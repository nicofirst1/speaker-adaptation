from src.commons import parse_args
from src.data.dataloaders import FinetuneDataset

episodes = 10
batch_size = 2
device = "cpu"
common_p = parse_args("sim")


def domain_difference():
    app_data = FinetuneDataset(
        "appliances",
        episodes * batch_size,
        device,
        common_p.vectors_file,
        common_p.img2dom_file,
    )

    food_data = FinetuneDataset(
        "food",
        episodes * batch_size,
        device,
        common_p.vectors_file,
        common_p.img2dom_file,
    )

    assert set(app_data.domain_images) & set(food_data.domain_images) == set()



if __name__ == "__main__":
    domain_difference()
