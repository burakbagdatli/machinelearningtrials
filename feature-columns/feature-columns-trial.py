import tensorflow as tf

tf.enable_eager_execution()

def _parse_csv_row(*vals):
    soil_type_t = tf.convert_to_tensor(vals[14:54])
    feat_vals = vals[:10] + (soil_type_t, vals[54])
    features = dict(zip(col_names, feat_vals))
    class_label = tf.argmax(vals[10:14], axis=0)
    return features, class_label
#
defaults = [tf.int32] * 55
dataset = tf.contrib.data.CsvDataset(['covtype.data'], defaults)
col_names = ['Elevation in meters',
             'Aspect in degrees azimuth',
             'Slope in degrees',
             'Horizontal distance in meters to nearest surface water features',
             'Vertical distance to nearest surface water features',
             'Horizontal distance in meters to nearest roadway',
             'Hillshade index at 9am, summer solstice',
             'Hillshade index at noon, summer solstice',
             'Hillshade index at 3pm, summer solstice',
             'Horizontal distance in meters to nearest wildfire ignition points',
             'Wilderness area designation',
             'Soil type designation',
             'Forest cover type designation']
dataset = dataset.map(_parse_csv_row).batch(64)
#print(list(dataset.take(1)))

numeric_features = [tf.feature_column.numeric_column(feat) for feat in col_names[:10]]
soil_type = tf.feature_column.numeric_column('Soil type designation', shape=(40,))
cover_type = tf.feature_column.categorical_column_with_identity('Forest cover type designation', num_buckets=8)

cover_embedding = tf.feature_column.embedding_column(cover_type, dimension=10)

columns = numeric_features + [soil_type, cover_embedding]
feature_layer = tf.keras.layers.DenseFeatures(columns)

model = tf.keras.Sequential([feature_layer,
                             tf.keras.layers.Dense(256),
                             tf.keras.layers.Dense(16),
                             tf.keras.layers.Dense(8),
                             tf.keras.layers.Dense(4, activation=tf.nn.softmax)])
#
