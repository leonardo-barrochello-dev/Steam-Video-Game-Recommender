import tensorflow as tf

class UserTower(tf.keras.Model):
    def __init__(self, embedding_dim=32):
        super().__init__()
        # In a real scenario, this would have embedding layers for user_id or complex feature combinations.
        # Here we just use the numerical features engineered (e.g. total_playtime_norm).
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.dense_out = tf.keras.layers.Dense(embedding_dim)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense_out(x)
        # L2 normalization for cosine similarity compatibility
        return tf.math.l2_normalize(x, axis=1)

class ItemTower(tf.keras.Model):
    def __init__(self, embedding_dim=32):
        super().__init__()
        # Similarly, item features like price_norm
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.dense_out = tf.keras.layers.Dense(embedding_dim)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense_out(x)
        # L2 normalization
        return tf.math.l2_normalize(x, axis=1)

class TwoTowerModel(tf.keras.Model):
    def __init__(self, embedding_dim=32):
        super().__init__()
        self.user_tower = UserTower(embedding_dim)
        self.item_tower = ItemTower(embedding_dim)

    def call(self, inputs):
        user_features = inputs['user_features']
        item_features = inputs['item_features']
        
        user_embeddings = self.user_tower(user_features)
        item_embeddings = self.item_tower(item_features)
        
        # Dot product
        dot_product = tf.reduce_sum(user_embeddings * item_embeddings, axis=1, keepdims=True)
        # Sigmoid to get probability [0, 1] for BinaryCrossentropy
        return tf.keras.activations.sigmoid(dot_product)

def build_and_compile_model(embedding_dim=32, learning_rate=0.001):
    model = TwoTowerModel(embedding_dim=embedding_dim)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )
    return model

if __name__ == "__main__":
    # Quick test of the model graph
    model = build_and_compile_model()
    # Dummy inputs for 51 features
    dummy_user = tf.random.normal((10, 51))
    dummy_item = tf.random.normal((10, 51))
    dummy_output = model({'user_features': dummy_user, 'item_features': dummy_item})
    print(f"Model test output shape: {dummy_output.shape}")
    # We can save weights using: model.save_weights('two_tower_weights')
