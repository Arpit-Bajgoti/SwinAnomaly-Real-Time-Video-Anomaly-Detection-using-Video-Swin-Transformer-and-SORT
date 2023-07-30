import pandas as pd

def create_csv(root_dir, number_of_frames = 4, frame_jump = 1):
    number_of_train_folders = len(next(os.walk(root_dir))[1])
    
    train_dataset = pd.DataFrame({'frames': [], 'label': []})
    for j in range(number_of_train_folders):
        lst = [
            f"Train{str(j+1).zfill(3)}/frame_{str(i+1).zfill(4)}.jpg" for i in range(0, len(fnmatch.filter(os.listdir(os.path.join(root_dir, f"Train{str(j+1).zfill(3)}")), '*.jpg')), frame_jump)]
        items = [(lst[i:i+number_of_frames], lst[i+number_of_frames]) for i in range(len(lst)-(number_of_frames+1))]
        x = pd.DataFrame(items, columns=["frames", "label"])
        train_dataset = train_dataset.append(x, ignore_index=True)
    return train_dataset