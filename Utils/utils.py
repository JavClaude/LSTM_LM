import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_sequence(model, tokenizer, seed: str, max_length: int):
    model.eval()
    model.to(device)

    SoftMax = torch.nn.Softmax(0)

    input = tokenizer.encode("<SOS> " + seed).ids
    output = input
    input = torch.tensor([input], dtype=torch.long, device=device)

    for _ in range(max_length):
        with torch.no_grad():
            hiddens = model.init_hiddens(1)

            logits, hiddens = model(
                input,
                hiddens
            )
            _, top = torch.topk(
                SoftMax(
                    logits[0][-1]
                ),
                1
            )

            input = top.unsqueeze(0)
            
    return output