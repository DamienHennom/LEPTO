from transform.transformation import GaussRankScaler

a = [10,12,15,30]
le = GaussRankScaler
print(le.fit_transform(a))
