from sklearn.model_selection import train_test_split

import os
import pandas as pd

class DataLoader():
    """
    Class to manage loading of data
    """
    # loading data and labels from directory
    def __init__(self, data_dir, labels_file, data_limit=None):
        """
        Initialise data loader class with relevant variables
        """
        self.train_dir = data_dir
        self.labels = pd.read_excel(labels_file)
        # additional label editing to make extraction easier
        self.labels["Subject"] = self.labels["Subject"].map(lambda x: x.replace(" ","_"))
        self.labels["CKD"] = self.labels["CKD"].astype(int)
        subjects = []
        for sub in self.labels["Subject"]:
            split_sub = sub.split("_")
            if len(split_sub) > 2:
                split_sub[2] = "00" + split_sub[2]
                subjects.append("_".join(split_sub[:3]))
            else:
                if len(split_sub[1]) < 2:
                   split_sub[1] = "0" + split_sub[1]
                   subjects.append("_".join(split_sub[:2]))
                else:
                    subjects.append(sub)
        self.labels["Subject"] = subjects
        additionalDf = pd.DataFrame(columns=self.labels.columns)
        # sort subjects with multiple scans
        multipleScanList = ["CKD_30", "CKD_1H_003", "CKD_1H_004", "CKD_1H_008", "HV_25", "HV_26", "HV_27", "HV_28", "HV_29"]
        for sub in self.labels["Subject"]:
            if sub in multipleScanList:
                numOfPatients = 5
                row = self.labels.loc[self.labels["Subject"] == sub]
                self.labels = self.labels.drop(row.index)
                for j in range(1, numOfPatients+1):
                    row.Subject = sub + "_R" + str(j)
                    additionalDf = additionalDf.append(row)
        self.labels = pd.concat([self.labels, additionalDf])
        self.labels = self.labels.sample(frac=1) # shuffle
        self.data_limit = data_limit
        self.labels = self.labels.reset_index(drop=True)
    
    def generate_partition(self):
        """
        Return dictionary containing data partitions for training, validation and testing
        """
        partition = {"train": None, "validation": None, "all": None}
        
        X_train, X_validation = train_test_split(self.labels["Subject"], test_size=0.2)
        
        partition["train"] = X_train.tolist()
        partition["validation"] = X_validation.tolist()
        partition["all"] = self.labels["Subject"].tolist()

        return partition

    def generate_labels(self, partition):
        """
        Return dictionary of IDs containing the labels for each ID
        """
        labels = {"train": None, "validation": None, "test": None}
        for set in partition.keys():
            for patient in partition[set]:
                mask = patient
                labels[set] = patient
        return labels

    def load_data(self):
        """
        Main function to load data
        """
        partition = self.generate_partition()
        labels = self.labels
        return partition, labels