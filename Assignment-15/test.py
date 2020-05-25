import visualize as viz
import hybrid_model as hm
import test_visualize as tv
from PIL import Image
from torchvision import datasets, models, transforms
def test(image_path, device):

  model = hm.Hybrid()
  model.to(device)

  model.eval()
  image = Image.open(image_path)
  data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49737222, 0.49328762, 0.46562814), (0.21714624, 0.21749634, 0.23051126)),
        ])

  image = data_transforms(image)

  seg, depth = model(image)

  tv.show(image, seg, depth)
  

  # with torch.no_grad():
  #   for data in test_loader:
  #     data["fg"] = data["fg"].to(device)
  #     data["bg"] = data["bg"].to(device)
  #     data["mask"] = data["mask"].to(device)
  #     output = model(data["fg"])

  #     test_loss += criterion(output, data["mask"], reduction='sum' ).item()
  #     pred = output.argmax(dim=1, keepdim=True)
  #     correct += pred.eq(target.view_as(pred)).sum().item()

  #     show(output.cpu(), nrow=2)
  #   test_loss /= len(test_loader.dataset)
  #   # viz.show(output.detach().cpu(), nrow=4)