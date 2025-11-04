import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps


radiomics_df = pd.read_csv(r"your features csv files")
cols_to_drop = [col for col in radiomics_df.columns if col.startswith("diagnostics_")] + ["Sample", "Freshness", "ImagePath"]
radiomics_features = radiomics_df.drop(columns=cols_to_drop)
labels = radiomics_df["Freshness"]
image_paths = radiomics_df["ImagePath"]

le = LabelEncoder()
y_encoded = le.fit_transform(labels)


# pre-VGG16（PyTorch）

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torchvision.models import vgg16, VGG16_Weights
weights = VGG16_Weights.IMAGENET1K_V1
vgg = vgg16(weights=weights)
model = nn.Sequential(vgg.features, vgg.avgpool)
model = model.to(device)
model.eval()


def resize_with_padding(img, target_size=224):
    old_size = img.size  # (width, height)
    ratio = float(target_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    img = img.resize(new_size, Image.LANCZOS)
    delta_w = target_size - new_size[0]
    delta_h = target_size - new_size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    new_img = ImageOps.expand(img, padding, fill=(0, 0, 0))
    return new_img

preprocess = transforms.Compose([
    transforms.Lambda(lambda img: resize_with_padding(img, target_size=450)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_dl_features(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"miage {img_path} fault: {e}")
        return None
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(input_batch)
    features = features.view(features.size(0), -1)
    return features.cpu().numpy().flatten()


dl_features_list = []
for path in image_paths:
    features = extract_dl_features(path)
    if features is None:
        features = np.zeros(25088)
    dl_features_list.append(features)
dl_features = np.array(dl_features_list)


scaler_radiomics = StandardScaler()
radiomics_scaled = scaler_radiomics.fit_transform(radiomics_features)

scaler_dl = StandardScaler()
dl_scaled = scaler_dl.fit_transform(dl_features)

# plot_lasso_path

def plot_lasso_path(X, y, feature_type, save_path):
    Cs = np.logspace(-3, 2, 20)
    coefs = []

    for c in Cs:
        lr = LogisticRegression(penalty='l1', solver='liblinear', C=c, max_iter=1000)
        lr.fit(X, y)
        coefs.append(lr.coef_.flatten())

    coefs = np.array(coefs)

    plt.figure(figsize=(10, 6))
    for i in range(coefs.shape[1]):
        plt.plot(np.log10(Cs), coefs[:, i], lw=1)

    plt.xlabel("log10(C)")
    plt.ylabel("LASSO-Coefficient")
    plt.title(f"{feature_type}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"{feature_type} LASSO PATH SAVE：{save_path}")
    plt.close()

def plot_selected_lasso_path(X, y, selector_model, feature_names, top_k, feature_type, save_path):
    mask = selector_model.get_support()
    selected_names = np.array(feature_names)[mask]
    selected_coefs = selector_model.estimator_.coef_.flatten()[mask]

    abs_sorted_idx = np.argsort(np.abs(selected_coefs))[::-1][:top_k]
    top_feat_names = selected_names[abs_sorted_idx]

    Cs = np.logspace(-3, 2, 20)
    coefs = []

    for c in Cs:
        lr = LogisticRegression(penalty='l1', solver='liblinear', C=c, max_iter=1000)
        lr.fit(X, y)
        full_coefs = lr.coef_.flatten()
        selected_full_coefs = full_coefs[mask]
        coefs.append(selected_full_coefs[abs_sorted_idx])

    coefs = np.array(coefs)

    plt.figure(figsize=(10, 6))
    for i in range(top_k):
        plt.plot(np.log10(Cs), coefs[:, i], label=top_feat_names[i])

    plt.xlabel("log10(C)")
    plt.ylabel("LASSO-Coefficient")
    plt.title(f"{feature_type} - Top {top_k} Feature Coefficient Paths")
    plt.legend(loc='best', fontsize=9)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"save top {top_k}  {feature_type} coefficient path diagram : {save_path}")
    plt.close()


def plot_combined_feature_bar(top_rad_feats, top_dl_feats, save_path):

    all_feats = [(f"Rad_{name}", coef) for name, coef in top_rad_feats] + \
                [(f"DL_{name}", coef) for name, coef in top_dl_feats]

    names = [item[0] for item in all_feats]
    values = [item[1] for item in all_feats]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(values)), values, tick_label=names)


    for bar, val in zip(bars, values):
        bar.set_color('blue' if val >= 0 else 'red')

    plt.xticks(rotation=45, ha='right')
    plt.ylabel("LASSO Coefficient")
    plt.title("Top 10 Radiomics + Top 10 DL Features")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved merged histogram : {save_path}")
    plt.close()


pipeline_rad = Pipeline([
    ('selector', SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear',
                                                     random_state=42, max_iter=1000),
                                 threshold='mean')),
    ('clf', LogisticRegression(max_iter=1000))
])
param_grid_rad = {'selector__estimator__C': [0.1, 1.0, 10, 100]}
grid_rad = GridSearchCV(pipeline_rad, param_grid_rad, cv=5, scoring='accuracy', n_jobs=-1)
grid_rad.fit(radiomics_scaled, y_encoded)
best_selector_rad = grid_rad.best_estimator_['selector']
radiomics_reduced = best_selector_rad.transform(radiomics_scaled)
print("Dimensions after dimensionality reduction :", radiomics_reduced.shape)

pipeline_dl = Pipeline([
    ('selector', SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear',
                                                     random_state=42, max_iter=1000),
                                 threshold='mean')),
    ('clf', LogisticRegression(max_iter=1000))
])
param_grid_dl = {'selector__estimator__C': [0.1, 1.0, 10, 100]}
grid_dl = GridSearchCV(pipeline_dl, param_grid_dl, cv=5, scoring='accuracy', n_jobs=-1)
grid_dl.fit(dl_scaled, y_encoded)
best_selector_dl = grid_dl.best_estimator_['selector']
dl_reduced = best_selector_dl.transform(dl_scaled)
print("Deep learning feature dimensionality reduction :", dl_reduced.shape)


def get_top_k_features(selector_model, feature_names, top_k=10):
    """
    selector_model
    feature_names
    top_k
    """
    mask = selector_model.get_support()
    selected_names = np.array(feature_names)[mask]
    selected_coefs = selector_model.estimator_.coef_.flatten()[mask]

    abs_sorted_idx = np.argsort(np.abs(selected_coefs))[::-1][:top_k]

    top_features = [(selected_names[i], selected_coefs[i]) for i in abs_sorted_idx]
    return top_features


top_rad_feats = get_top_k_features(best_selector_rad, radiomics_features.columns, top_k=10)
print("\ntop 10 of radiomics features：")
for name, coef in top_rad_feats:
    print(f"{name}: {coef:.4f}")

dl_feat_names = [f"DL_{i}" for i in range(dl_features.shape[1])]
top_dl_feats = get_top_k_features(best_selector_dl, dl_feat_names, top_k=10)
print("\ntop 10 of deep learning features：")
for name, coef in top_dl_feats:
    print(f"{name}: {coef:.4f}")


# draw lasso path
plot_lasso_path(radiomics_scaled, y_encoded, "Radiomics", "E:\lasso-test\lasso_path_radiomics.png")
plot_lasso_path(dl_scaled, y_encoded, "Deep Learning", "E:\lasso-test\lasso_path_dl.png")
plot_combined_feature_bar(top_rad_feats, top_dl_feats, "E:/lasso-test/zhuzhuang_combined.png")


plot_selected_lasso_path(
    radiomics_scaled, y_encoded, best_selector_rad, radiomics_features.columns,
    top_k=10, feature_type="Radiomics", save_path="E:/lasso-test/selected_lasso_radiomics.png"
)

plot_selected_lasso_path(
    dl_scaled, y_encoded, best_selector_dl, dl_feat_names,
    top_k=10, feature_type="Deep Learning", save_path="E:/lasso-test/selected_lasso_dl.png"
)


combined_features = np.concatenate([radiomics_reduced, dl_reduced], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    combined_features, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)


clf = SVC(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("accuracy score of testset:", accuracy_score(y_test, y_pred))
print("classification report of testset:\n", classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

from sklearn.metrics import confusion_matrix
#TN, FP, FN, TP
cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

specificity = TN / (TN + FP + 1e-8)
print(f"Specificity of testset: {specificity:.4f}")


def compute_net_benefit(y_true, y_prob, thresholds):
    N = len(y_true)
    results = []

    for thr in thresholds:
        pred_positive = y_prob >= thr
        TP = np.sum((pred_positive == 1) & (y_true == 1))
        FP = np.sum((pred_positive == 1) & (y_true == 0))
        NB = (TP / N) - (FP / N) * (thr / (1 - thr))
        results.append(NB)
    return results


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


thresholds = np.linspace(0.01, 0.99, 99)

# Net Benefit
def compute_net_benefit(y_true, y_prob, thresholds):
    N = len(y_true)
    results = []
    for thr in thresholds:
        pred_pos = (y_prob >= thr)
        TP = np.sum((pred_pos == 1) & (y_true == 1))
        FP = np.sum((pred_pos == 1) & (y_true == 0))
        nb = (TP / N) - (FP / N) * (thr / (1 - thr))
        results.append(nb)
    return results

# 1) SVM
svm_clf = SVC(probability=True, random_state=42)
svm_clf.fit(X_train, y_train)
nb_svm = compute_net_benefit(y_test, svm_clf.predict_proba(X_test)[:, 1], thresholds)

# 2) XGBoost
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
xgb_clf.fit(X_train, y_train)
nb_xgb = compute_net_benefit(y_test, xgb_clf.predict_proba(X_test)[:, 1], thresholds)

# 3) Logistic Regression
lr_clf = LogisticRegression(max_iter=1000)
lr_clf.fit(X_train, y_train)
nb_lr = compute_net_benefit(y_test, lr_clf.predict_proba(X_test)[:, 1], thresholds)

# 4) Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
nb_rf = compute_net_benefit(y_test, rf_clf.predict_proba(X_test)[:, 1], thresholds)

# 5) LightGBM
lgbm_clf = LGBMClassifier(n_estimators=100, random_state=42)
lgbm_clf.fit(X_train, y_train)
nb_lgbm = compute_net_benefit(y_test, lgbm_clf.predict_proba(X_test)[:, 1], thresholds)

# Treat-All / Treat-None
event_rate = y_test.mean()
nb_all = [event_rate - (1 - event_rate) * thr / (1 - thr) for thr in thresholds]
nb_none = [0] * len(thresholds)


plt.figure(figsize=(10, 6))
plt.plot(thresholds, nb_svm, label="SVM", color="blue", linewidth=2)
plt.plot(thresholds, nb_xgb, label="XGBoost", color="orange", linewidth=2)
plt.plot(thresholds, nb_lr, label="Logistic Regression", color="green", linewidth=2)
plt.plot(thresholds, nb_rf, label="Random Forest", color="purple", linewidth=2)
plt.plot(thresholds, nb_lgbm, label="LightGBM", color="red", linewidth=2)
plt.plot(thresholds, nb_all, label="Treat-All", linestyle="--", color="gray", linewidth=2)
plt.plot(thresholds, nb_none, label="Treat-None", linestyle="--", color="black", linewidth=2)

plt.xlabel("Threshold Probability")
plt.ylabel("Net Benefit")
plt.title("Decision Curve Analysis")
plt.ylim(-0.5, 1.0)
plt.legend(loc="upper right")
plt.grid(linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("E:/lasso-test/dca_compare_extended3.png", dpi=300)
plt.close()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


models = {
    'SVM': SVC(kernel='rbf', probability=True, gamma='auto', random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'LightGBM': LGBMClassifier(n_estimators=100, random_state=42)
}

calibration_data = {}
for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    calibration_data[name] = (prob_pred, prob_true)

plt.figure(figsize=(8, 6))
plt.plot([0,1], [0,1], 'k--', label='Perfect Calibration')
for name, (prob_pred, prob_true) in calibration_data.items():
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label=name)

plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curves Comparison')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("E:/lasso-test/calibration_curves_extended2.png", dpi=300)
plt.close()
