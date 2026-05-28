import tensorflow as tf

FEATURE_DIM = 51


def build_and_compile_model(embedding_dim=32, learning_rate=0.001):
    user_tower = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(embedding_dim),
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)),
    ], name='user_tower')

    item_tower = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(embedding_dim),
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)),
    ], name='item_tower')

    user_input = tf.keras.Input(shape=(FEATURE_DIM,))
    item_input = tf.keras.Input(shape=(FEATURE_DIM,))

    user_emb = user_tower(user_input)
    item_emb = item_tower(item_input)

    dot_product = tf.keras.layers.Dot(axes=1)([user_emb, item_emb])
    output = tf.keras.layers.Activation('sigmoid')(dot_product)

    model = tf.keras.Model(inputs=[user_input, item_input], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )
    return model


if __name__ == "__main__":
    model = build_and_compile_model()
    dummy_user = tf.random.normal((10, FEATURE_DIM))
    dummy_item = tf.random.normal((10, FEATURE_DIM))
    dummy_output = model([dummy_user, dummy_item])
    print(f"Model test output shape: {dummy_output.shape}")
