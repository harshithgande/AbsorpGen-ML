from data_loader import load_mock_data, preprocess_data

df = load_mock_data()
print("Original Data:")
print(df)

X, y = preprocess_data(df)
print("\nProcessed Features Shape:", X.shape)
print("Target Values:", y.tolist())
