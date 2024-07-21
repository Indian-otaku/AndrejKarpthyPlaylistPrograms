class MeanSquaredError:
    def __call__(self, y_true, y_pred):
        n = len(y_true)
        return sum([(yp_i[0] - yt_i[0])**2 for yt_i, yp_i in zip(y_true, y_pred)]) / n
    