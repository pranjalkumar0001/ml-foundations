import torch
optim = torch.optim
tensor = torch.tensor
batch_size = 32
channels = 64
time_steps = 500
X = torch.randn(batch_size, channels, time_steps)
X_rnn = X.permute(0,2,1)
# print(X.shape)
# print(X_rnn.shape)
rnn_layer = torch.nn.RNN(input_size=64, hidden_size=128, batch_first=True)
output, final_hidden_state = rnn_layer(X_rnn)
print("output shape is ", output.shape)
print('hidden_state shape is', final_hidden_state.shape)
classifier = torch.nn.Linear(in_features=128, out_features=4)
flattened_summary = final_hidden_state.squeeze(0)
final_prediction = classifier(flattened_summary)
print("final Prediction shape:", final_prediction.shape)

Y_true = torch.randint(0,4,(32,))
loss_fn = torch.nn.CrossEntropyLoss()
all_paramerters = list(rnn_layer.parameters()) + list(classifier.parameters())
optimiser = optim.Adam(all_paramerters, lr=0.001)
print("\n\n ----starting training ----")
for epoch in range(100):
    optimiser.zero_grad()
    output, final_hidden = rnn_layer(X_rnn)
    prediction = classifier(final_hidden.squeeze(0))
    loss = loss_fn(prediction, Y_true)
    loss.backward()
    optimiser.step()

    if epoch%20 == 0 or epoch == 99:
        print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f}")