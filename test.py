test_dir = "/home/tinkerspace/Abhinand/plant_dataset/dataset/test"
test = ImageFolder(test_dir, transform=transforms.ToTensor())

# predicting first image
img, label = test[0]
plt.imshow(img.permute(1, 2, 0))
print('Label:', test_images[0], ', Predicted:', predict_image(img, model))