import pandas as pd


class AverageMeter:
    def __init__(self):
        self.sum = 0.
        self.num = 0

    def reset(self):
        self.sum = 0.
        self.num = 0

    def update(self, value, num=1):
        self.sum += value
        self.num += num

    def avg(self):
        return self.sum / self.num


def mp_to_landmark(landmarks):
    landmark_list = []
    for lm in landmarks:
        landmark_list.append([lm.x, lm.y, lm.z])

    return landmark_list


def absolute_points_to_relative(points):
    base_x, base_y, base_z = points[0]
    points = list(map(lambda p: [p[0] - base_x, p[1] - base_y, p[2] - base_z], points))
    features = []
    for p in points:
        features += p

    return features


def append_to_df(df, value, label):
    columns = list(df.columns)
    print(columns)
    value.append(label)
    new_df = pd.DataFrame([value], columns=columns)
    df = df.append(new_df, ignore_index=True)
    print(df)

    return df
