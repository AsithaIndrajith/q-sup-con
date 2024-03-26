import torch as nn

def get_accuracy( data_loder, device, model ):
    # Make predictions
    y_pred = []
    y_pred_2 = []
    test_accuracy = 0
    with nn.no_grad():
        for inputs, labels in data_loder:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            y_pred_2.extend(outputs)
            _, predicted_labels = nn.max(outputs, dim=1)
            y_pred.extend(predicted_labels.tolist())
            test_accuracy += (predicted_labels == labels).sum().item()  # Count correctly predicted samples

    test_accuracy = 100.0 * test_accuracy / len(data_loder.dataset)
    return test_accuracy, y_pred, y_pred_2