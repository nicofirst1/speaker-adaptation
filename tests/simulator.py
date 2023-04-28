import torch

from src.models.simulator.SimulatorModel import SimulatorModel

device = "cpu"
batch_size = 2
img_dim = 2048
vocab_size = 6971
utt_size = 8
embedding_dim = 16

sim_model = SimulatorModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_dim=16,
    img_dim=img_dim,
    att_dim=16,
    dropout_prob=0,
    domain="food",
    device=device,
).to(device)

################
# Datapoint
################
separate_images = torch.randn(batch_size, 6, img_dim).to(device)
utterance = torch.randint(0, vocab_size, (batch_size, utt_size)).to(device)
masks = torch.zeros(batch_size, utt_size, 1).to(device)
speaker_embeds = torch.randn(batch_size, embedding_dim).to(device)
target = torch.randint(0, 6, (batch_size, 1)).to(device)


def test_only_utt_update():
    sim_model.train()

    # define optimizer
    optimizer = torch.optim.Adam(sim_model.parameters(), lr=0.001)

    # clone named parameters
    old_params = [(p[0], p[1].clone()) for p in sim_model.named_parameters()]
    old_params = dict(old_params)

    sim_out = sim_model(
        separate_images,
        utterance,
        masks,
    )

    # estimate loss with cross entropy
    loss = torch.nn.functional.cross_entropy(sim_out, target)
    loss.backward()
    optimizer.step()

    new_params = dict(sim_model.named_parameters())

    # check that the parameters have changed
    same_keys = []
    for p in old_params:
        if torch.equal(old_params[p], new_params[p]):
            same_keys.append(p)

    should_be_same = [
        "lin_emb2hid_emb.1.weight",
        "lin_emb2hid_emb.1.bias",
        "lin_mm_emb.1.weight",
        "lin_mm_emb.1.bias",
    ]
    assert (
        same_keys == should_be_same
    ), f"Parameters {same_keys} should be the same as {should_be_same}"


def test_all_update():
    sim_model.train()

    # define optimizer
    optimizer = torch.optim.Adam(sim_model.parameters(), lr=0.001)

    # clone named parameters
    old_params = [(p[0], p[1].clone()) for p in sim_model.named_parameters()]
    old_params = dict(old_params)

    sim_out = sim_model(
        separate_images,
        utterance,
        masks,
        speaker_embeds=speaker_embeds,
    )

    # estimate loss with cross entropy
    loss = torch.nn.functional.cross_entropy(sim_out, target)
    loss.backward()
    optimizer.step()

    new_params = dict(sim_model.named_parameters())

    # check that the parameters have changed
    same_keys = []
    for p in old_params:
        if torch.equal(old_params[p], new_params[p]):
            same_keys.append(p)

    assert same_keys == [], f"No parameters should be the same"


def test_utt_freeze():
    sim_model.train()

    # define optimizer
    optimizer = torch.optim.Adam(sim_model.parameters(), lr=0.001)
    sim_model.freeze_utts_stream()


    # clone named parameters
    old_params = [(p[0], p[1].clone()) for p in sim_model.named_parameters()]
    old_params = dict(old_params)

    sim_out = sim_model(
        separate_images,
        utterance,
        masks,
        speaker_embeds=speaker_embeds,
    )

    # estimate loss with cross entropy
    loss = torch.nn.functional.cross_entropy(sim_out, target)
    loss.backward()
    optimizer.step()

    new_params = dict(sim_model.named_parameters())

    # check that the parameters have changed
    same_keys = []
    for p in old_params:
        if torch.equal(old_params[p], new_params[p]):
            same_keys.append(p)

    same_keys=[x for x in same_keys if "emb" in x]
    assert same_keys == [], f"No parameters should be the same"

if __name__ == "__main__":
    test_only_utt_update()
    sim_model.zero_grad()
    test_all_update()
    sim_model.zero_grad()
    test_utt_freeze()
    sim_model.zero_grad()

