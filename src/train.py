from src.model import model
from src.dataset import dataset


def train(
    train_path,
    aug_train_path,
    val_path,
    save_path="src/models",
    seed=42,
    standard_size=(64, 64),
    label_size_factor=1,
    max_iter=10000,
):
    train_data = dataset(
        img_dir=train_path,
        augment_path=aug_train_path,
        label_size_factor=label_size_factor,
        standard_size=standard_size,
        seed=seed,
    )
    val_data = dataset(val_path, train=False, standard_size=standard_size)
    my_model = model(seed, standard_size=standard_size)
    my_model.train_svm(train_data, verbose=0, max_iter=max_iter)
    print(
        f"Model trained with {my_model.name} with seed {seed}, max_iter {max_iter}, standard_size {standard_size}, label_size_factor {label_size_factor}"
    )
    accuracy = my_model.evaluate(val_data)
    print(f"Validation accuracy with {my_model.name}: {accuracy}")
    my_model.save(save_path)
    return my_model
