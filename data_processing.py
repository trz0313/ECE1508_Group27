import os
import random
import shutil
import pydicom

def get_liver_list():
    """
    Generate lists of liver MRI scan files for training, validation, and testing.
    It processes directories, handles mismatched file counts, and ensures data consistency.
    """
    base_path = "../../../data/MRI/Liver/"  # Base path for liver MRI data
    output_path = "../../../data/catch"

    # Create the output directory if it doesn't exist
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # Collect all subdirectories containing "SE0"
    path_list = []
    for root, dirs, files in os.walk(base_path, topdown=False):
        if "SE0" in root:
            path_list.append(root)
    random.shuffle(path_list)

    # Open output files for train, validation, and test lists
    train_file = open("train_liver.txt", "w")
    val_file = open("val_liver.txt", "w")
    test_file = open("test_liver.txt", "w")

    processed_ids = []
    for i, sub_path in enumerate(path_list):
        try:
            # Ensure matching directories are consistent
            paired_path = sub_path.replace("SE0", "SE1")
            data_files = os.listdir(sub_path)
            target_files = os.listdir(paired_path)

            # Handle mismatched file counts by renaming or removing directories
            if len(data_files) != len(target_files):
                if len(data_files) % len(target_files) == 0:
                    os.rename(sub_path, sub_path.replace("SE0", "SE_TEMP"))
                    os.rename(paired_path, sub_path)
                    os.rename(sub_path.replace("SE0", "SE_TEMP"), paired_path)

            # Filter files and verify consistency
            for file_name in data_files:
                file_path = os.path.join(sub_path, file_name)
                paired_file_path = file_path.replace("SE0", "SE1")
                if not os.path.exists(paired_file_path):
                    os.remove(file_path)
                    continue

                # Read DICOM headers for validation
                dsA = pydicom.dcmread(file_path, force=True)
                dsB = pydicom.dcmread(paired_file_path, force=True)

                # Patient ID validation
                if dsA.PatientID not in processed_ids:
                    processed_ids.append(dsA.PatientID)
                elif processed_ids[-1] != dsA.PatientID:
                    shutil.rmtree(sub_path.split("/ST0")[0])
                    break

                # Validate DICOM properties specific to liver MRI
                if dsA.RescaleIntercept != -1024 or dsB.RescaleIntercept != -1024:
                    continue
                if dsA.AccessionNumber != dsB.AccessionNumber:
                    continue
                if dsA.SliceLocation != dsB.SliceLocation:
                    continue

                # Validate protocol for liver scans (e.g., hepatobiliary phase)
                if "Liver" not in dsA.ProtocolName and "Liver" in dsB.ProtocolName:
                    temp_path = os.path.join(output_path, file_path.split("/")[-1])
                    shutil.move(file_path.replace("SE0", "SE1"), temp_path)
                    shutil.move(file_path, file_path.replace("SE0", "SE1"))
                    shutil.move(temp_path, file_path)
                else:
                    continue

            # Split data into training, validation, and testing
            if i <= int(len(path_list) * 0.6):
                train_file.writelines(file_path + "\n")
            elif i <= int(len(path_list) * 0.8):
                val_file.writelines(file_path + "\n")
            else:
                test_file.writelines(file_path + "\n")

        except Exception as e:
            print(f"Error processing {sub_path}: {e}")
            continue

    train_file.close()
    val_file.close()
    test_file.close()
    print(f"Processed IDs: {len(processed_ids)}")
