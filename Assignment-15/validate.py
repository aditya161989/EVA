import visualize as viz
import torch

def validate(model, criterion, device, validate_loader):
    batches = 0
    loss = 0.0
    with torch.no_grad():
        for batchidx, data in enumerate(validate_loader):
            data["fg"] = data["fg"].to(device)
            data["bg"] = data["bg"].to(device)
            data["mask"] = data["mask"].to(device)
            output = model(data["fg"])
            batches +=1

            loss += criterion(output, data["mask"])

    return loss
#     viz.show(output.detach().cpu(),data["mask"].detach().cpu())