import tqdm
import torch
import mlflow

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_model(model: torch.nn.Module, train_iterator, criterion, optimizer, max_grad_norm, global_train_it):
    model.train()
    model.zero_grad()

    epoch_loss = 0
    local_train_it = 0

    hiddens = model.init_hiddens(train_iterator.batch_size)

    for i in tqdm.tqdm(range(0, len(train_iterator), train_iterator.bptt), desc="Training..."):
        batch = train_iterator.get_batches(i)
        batch = tuple(t.to(device) for t in batch)

        seq_train, seq_target = batch
        logits, hiddens = model(seq_train, hiddens)
        hiddens = tuple(t.detach() for t in hiddens) # Do not backprop throught hiddens state

        loss = criterion(logits.transpose(2, 1), seq_target)
        loss.backward()

        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        model.zero_grad()

        # Mlflow tracking metric #
        mlflow.log_metric("training loss", loss.item(), step=global_train_it)
        global_train_it += 1

        epoch_loss += loss.item()
        local_train_it += 1

    epoch_loss /= local_train_it

    return epoch_loss, global_train_it



def eval_model(model, test_iterator, criterion, global_eval_it):
    model.eval()

    epoch_loss = 0
    local_eval_it = 0

    with torch.no_grad():
        hiddens = model.init_hiddens(test_iterator.batch_size)

        for i in tqdm.tqdm(range(0, len(test_iterator), test_iterator.bptt), desc="Testing..."):
            batch = test_iterator.get_batches(i)
            batch = tuple(t.to(device) for t in batch)

            seq_train, seq_target = batch
            logits, hiddens = model(seq_train, hiddens)
            hiddens = tuple(t.detach() for t in hiddens) # Do not backprop throught hiddens state

            loss = criterion(logits.transpose(2, 1), seq_target)
        
            # Mlflow tracking metric
            mlflow.log_metric("eval loss", loss.item(), step=global_eval_it)
            global_eval_it += 1
            epoch_loss += loss.item()
            local_eval_it += 1
        
    epoch_loss /= local_eval_it

    return epoch_loss, global_eval_it
