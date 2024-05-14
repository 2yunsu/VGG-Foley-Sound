import pandas as pd

vggsound = pd.read_csv('vggsound.csv')
vggsound.columns = ["YouTube ID", "start seconds", "label", "train/test split"]
vggsound['thumbnail'] = 'https://img.youtube.com/vi/' + vggsound['YouTube ID'] + '/0.jpg'

labels = pd.read_csv('AStest_refined.csv')
labels.columns = ["label", "action", "material", "many"]
labels = labels.drop(columns = ["action", "material", "many"])
labels = labels.values.tolist()
labels_list = []
for element in labels:
    labels_list += element
print(labels_list)
refine = vggsound[vggsound['label'].isin(labels_list)]
# print(labels.head())
print(refine['label']=='subway, metro, underground')
refine.to_csv("extracted_VGGSound.csv", index=False)