import logging

def log_train(message, epoch, train_loss, train_acc, val_loss, val_acc, log_file):
    logging.basicConfig(filename=log_file, level=logging.INFO, format='Epoch %(epoch)d - train loss: %(train_loss).4f, train acc: %(train_acc).4f, val loss: %(val_loss).4f, val acc: %(val_acc).4f')
    logging.info(message, extra={'epoch': epoch, 'train_loss':train_loss, 'train_acc':train_acc, 'val_loss':val_loss, 'val_acc':val_acc})

def log_test(msg, loss, acc, log_file):
    logging.basicConfig(filename=log_file, level=logging.INFO, format='Testing - loss: %(loss).4f, acc: %(acc).4f')
    logging.info(msg, extra={'loss':loss, 'acc':acc})

if __name__ == '__main__':
    for epoch in range(1, 6):
        # log_train("Epoch completed.", epoch, 0.12, 0.12, 0.12, 0.12, "./output/log.log")
        log_test("VGG16", 0.12, 0.12, "./output/log.log")
