from radiomics import featureextractor as FEE
import os
import pandas as pd

file_path = 'E:/you image/'
para_path = 'D:/your params.yaml'

extractor = FEE.RadiomicsFeatureExtractor(parameter_file=para_path)
extractor.settings['force2D'] = True

print("Extraction parameters:\n\t", extractor.settings)
print("Enabled filters:\n\t", extractor.enabledImagetypes)
print("Enabled features:\n\t", extractor.enabledFeatures)

df_samples = pd.DataFrame()
count_success = 0
for root, dirs, files in os.walk(file_path):
    file_pairs = {}
    for file in files:
        current_path = os.path.join(root, file)
        if file.endswith("image.nrrd"):
            key = file.replace("_gray_image.nrrd", "")
            file_pairs.setdefault(key, {})['image'] = current_path
        elif file.endswith("label.nrrd"):
            key = file.replace("_label.nrrd", "")
            file_pairs.setdefault(key, {})['label'] = current_path
        print(current_path)
    for key, pair in file_pairs.items():
        if 'image' in pair and 'label' in pair:
            ori_path = pair['image']
            lab_path = pair['label']
            try:
                result = extractor.execute(ori_path, lab_path)
                count_success += 1
            except Exception as e:
                print(f"Error processing {ori_path} and {lab_path}: {e}")
                continue
            print("Result type:", type(result))
            print("Calculated features:")
            for k, v in result.items():
                print("\t", k, ":", v)
            sample_name = key
            png_path = os.path.join(os.path.dirname(ori_path), sample_name + ".png")
            png_path = os.path.normpath(png_path).replace('\\', '/')

            if "fresh" in sample_name.lower():
                freshness = "fresh"
            elif "old" in sample_name.lower():
                freshness = "old"
            else:
                freshness = "unknown"
            sample_features = {
                "Sample": sample_name,
                "Freshness": freshness,
                "ImagePath": png_path
            }
            for k, v in result.items():
                sample_features[k] = v

            df_samples = pd.concat([df_samples, pd.DataFrame([sample_features])], ignore_index=True)

output_path = r'D:\Users\DSY\PycharmProjects\pythonProject/test-all_features.csv'
df_samples.to_csv(output_path, index=False)
print("saved")
print(f"number of samples successfully extracted and saved：{count_success}")




