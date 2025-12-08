from keras.utils import Sequence
import numpy as np

class CampaignSequenceGenerator(Sequence):
    def __init__(self, df, features, target_col, seq_len=SEQ_LEN, horizon=HORIZON, batch_size=batch_size, shuffle=True):
        self.df = df
        self.features = features
        self.target_col = target_col
        self.seq_len = seq_len
        self.horizon = horizon
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Campaign groups
        self.campaign_groups = {c: g.sort_values('date') for c, g in df.groupby('traffic_source_campaign_name_anon')}
        
        # Indices for sequences
        self.indices = []
        for c, g in self.campaign_groups.items():
            max_start = len(g) - seq_len - horizon + 1
            if max_start > 0:
                self.indices.extend([(c, i) for i in range(max_start)])
        
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch, y_batch, c_batch = [], [], []

        for campaign, start_idx in batch_indices:
            g = self.campaign_groups[campaign]
            X_seq = g[self.features].values[start_idx:start_idx + self.seq_len].astype('float32')
            y_seq = g[self.target_col].values[start_idx + self.seq_len:start_idx + self.seq_len + self.horizon].astype('float32')
            X_batch.append(X_seq)
            y_batch.append(y_seq)
            c_batch.append(g['campaign_id'].values[start_idx + self.seq_len])

        # Return a tuple (features, target)
        return (np.array(X_batch), np.array(c_batch)), np.array(y_batch)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)