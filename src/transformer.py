import datetime
import time
import os

import numpy as np
from IPython import display
import tensorflow as tf

from utils import generate_images, Generator, Discriminator, LOGS_FIT_DIR

LAMBDA = 100
STEP_SIZE = 1000
checkpoint_dir = "./training_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")


def generator_loss(disc_generated_output, gen_output, target):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # Mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


# loss function
def discriminator_loss(disc_real_output, disc_generated_output):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(
        tf.zeros_like(disc_generated_output), disc_generated_output
    )

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


@tf.function
def train_step(
    generator,
    discriminator,
    generator_optimizer,
    discriminator_optimizer,
    summary_writer,
    input_image,
    target,
    step,
):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
            disc_generated_output, gen_output, target
        )
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(
        gen_total_loss, generator.trainable_variables
    )

    discriminator_gradients = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(generator_gradients, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(discriminator_gradients, discriminator.trainable_variables)
    )

    with summary_writer.as_default():
        tf.summary.scalar("gen_total_loss", gen_total_loss, step=step // 20)
        tf.summary.scalar("gen_gan_loss", gen_gan_loss, step=step // 20)
        tf.summary.scalar("gen_l1_loss", gen_l1_loss, step=step // 20)
        tf.summary.scalar("disc_loss", disc_loss, step=step // 20)


def train_and_fit(
    train_ds,
    test_ds,
    steps=STEP_SIZE,
):
    generator = Generator()
    generator.summary()

    discriminator = Discriminator()
    discriminator.summary()

    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator,
    )

    summary_writer = tf.summary.create_file_writer(
        LOGS_FIT_DIR + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    example_input, example_target = next(iter(test_ds.take(1)))
    start = time.time()

    for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
        if (step) % 20 == 0:
            display.clear_output(wait=True)

            if step != 0:
                print(f"Time taken for 20 steps: {time.time()-start:.2f} sec\n")

            start = time.time()

            generate_images(
                generator,
                example_input,
                example_target,
                "train_and_fit_" + str(int(step)),
            )
            print(f"Step: {step//20}k")

        train_step(
            generator,
            discriminator,
            generator_optimizer,
            discriminator_optimizer,
            summary_writer,
            input_image,
            target,
            step,
        )

        # Training step
        if (step + 1) % 20 == 0:
            print(".", end="", flush=True)

        # Save (checkpoint) the model every 5k steps
        if (step + 1) % 1000 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
