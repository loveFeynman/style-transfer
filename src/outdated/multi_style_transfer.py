def train_vae():
    pass

def is_vae_trained():
    pass

def get_trained_vae():
    return None, [], [[]] #returns the output of the VAE, the latent vector, and the positions on the latent vector that represent each class

def preprocess_style_weights(weights):
    sum_weights = sum(weights)
    return [x/sum_weights for x in weights]

def feed(a, b):
    pass

def resolve(a):
    pass

if __name__=='__main__':

    if not is_vae_trained():
        train_vae()
    else:
        vae_output, vae_latent_vector, class_vectors = get_trained_vae()

        style_network_image_input = None
        style_network_style_weights_input = None
        style_network_output = None

        loss_network_style_input = None
        loss_network_content_input = None
        loss_network_stylized_output = None

        loss_network_style_loss = None
        loss_network_content_loss = None
        loss_network_total_loss = None

        optimizer = None

        TRAINING_EPOCHS = 1
        STEPS_PER_EPOCH = 1000


        for epoch in range(TRAINING_EPOCHS):

            content_images = []

            for step in range(STEPS_PER_EPOCH):

                vae_latent_vector_sample = None #randomly pick a position on the latent vector
                feed(vae_latent_vector_sample, vae_latent_vector)
                vae_style_sample = resolve(vae_output) #sample the vae at that position

                feed(preprocess_style_weights(vae_latent_vector_sample), style_network_style_weights_input)
                feed(content_images[step], style_network_image_input)
                stylized_image = resolve(style_network_output)

                feed(vae_style_sample, loss_network_style_input)
                feed(content_images[step], loss_network_content_input)
                feed(stylized_image, loss_network_style_input)
                loss = resolve(loss_network_total_loss)

                resolve(optimizer)