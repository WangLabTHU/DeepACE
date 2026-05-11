
from deepDNAshape import predictor
import numpy as np
import pandas as pd

## Digital_Platform_deepdnashape

class dDNAshape():
    def __init__(self, layer=4, batch_size=2048, mode="cpu"):
        
        self.layer = layer
        self.batch_size = batch_size
        self.mode = mode
        self.pretrained_model = predictor.predictor(mode = mode)

    def predict_on_feature(self, seqs, feature):
        
        seqBatch = []
        pred_list = []
        for seq in seqs:
            seqBatch.append(seq.strip())
            if len(seqBatch) == self.batch_size:
                pred = self.pretrained_model.predictBatch(feature, seqBatch, self.layer)
                pred_list += pred.tolist()
        
        if seqBatch != []:
            pred = self.pretrained_model.predictBatch(feature, seqs, self.layer)
            pred_list += pred.tolist()

        # pred_list =  [arr.tolist() for arr in pred_list]
        return pred_list

    def predict(self, seqs, csv=False):
        feature_list = ["Shift", "Slide", "Rise", "Tilt", "Roll", "HelT", # interbase_features
                        "Shear", "Stretch", "Stagger", "Buckle", "ProT", "Opening", # intrabase_features
                        "MGW", "EP"] 
        
        pred_all_features = []
        for feature in feature_list:
            pred_list = self.predict_on_feature(seqs, feature)
            pred_all_features.append(pred_list) # (num_features, num_seqs, feature_len)
        
        pred_all_features_groupby_seqs = []
        for i in range(len(seqs)):
            tmp = []
            for j in range(len(feature_list)):
                tmp.append(pred_all_features[j][i])
            pred_all_features_groupby_seqs.append(tmp)
        # (num_seqs, num_features, feature_len)
        
        df = []
        if csv:
            df = pd.DataFrame({"seqs": seqs, 
                               **{feature: [preds[j] for preds in pred_all_features_groupby_seqs] for j, feature in enumerate(feature_list)}})
        
        return pred_all_features_groupby_seqs, df
        
        
    def quick_valid(self):
        # https://github.com/JinsenLi/deepDNAshape/tree/main?tab=readme-ov-file#predict-any-dna-shape-from-a-fasta-sequence-file
        
        seqs = [
            "ACGTAAAAGGGGATAACCG",
            "CCGTAGGG",
            "GGTGAGGGGGGGGGGGGGG"
        ]
        _, df = self.predict(seqs, csv=True)
        df.to_csv("./valids/DeepDNAshape/val.csv")
        
        print(df.loc[:,"MGW"])
        mgw_list = list(df.loc[:,"MGW"])
        if all(round(x[0], 6) == y for x, y in zip(mgw_list[:3], [5.335149, 4.879819, 4.977294])):
            print("The current outputs are in alignment with github examples (https://github.com/JinsenLi/deepDNAshape/tree/main?tab=readme-ov-file#predict-any-dna-shape-from-a-fasta-sequence-file).")
            
    def predict_bash(self, seqs, csv_features, mode="total"):
        
        feature_list = ["Shift", "Slide", "Rise", "Tilt", "Roll", "HelT", # interbase_features
                        "Shear", "Stretch", "Stagger", "Buckle", "ProT", "Opening", # intrabase_features
                        "MGW", "EP"] 
        
        pred_all_features = []
        for feature in feature_list:
            pred_list = self.predict_on_feature(seqs, feature)
            pred_all_features.append(pred_list) # (num_features, num_seqs, feature_len)
        
        pred_all_features_groupby_seqs = []
        for i in range(len(seqs)):
            tmp = []
            for j in range(len(feature_list)):
                tmp.append(pred_all_features[j][i])
            pred_all_features_groupby_seqs.append(tmp)
        # (num_seqs, num_features, feature_len)
        
        pred_list = pred_all_features_groupby_seqs
        
        if mode == "center":
            pred_deepdnashape = []

            seq_len = len(seqs[0])
            for i in range(len(pred_all_features_groupby_seqs)):
                tmp = []
                # interbase_features
                for j in range(6):
                        out_len = seq_len - 1
                        mid_start = (out_len - 1) // 2
                        mid_end = mid_start + 2 - (out_len % 2)
                        preds = np.mean(pred_all_features_groupby_seqs[i][j][mid_start:mid_end], axis=-1)
                        tmp.append(preds)

                # intrabase_features
                for j in range(6, 14):
                    out_len = seq_len
                    mid_start = (out_len - 1) // 2
                    mid_end = mid_start + 2 - (out_len % 2)
                    preds = np.mean(pred_all_features_groupby_seqs[i][j][mid_start:mid_end], axis=-1)
                    tmp.append(preds)
                pred_deepdnashape.append( tmp )

            pred_list = np.array(pred_deepdnashape)
        
        anno_df = pd.read_csv(csv_features)
        anno_df = anno_df[anno_df["model"] == "DeepDNAshape"]
        anno_df = anno_df.drop('Unnamed: 0', axis=1).reset_index()
        return pred_list, anno_df
            