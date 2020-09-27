import pointnetvlad_cls
import tensorflow as tf


# Volume loss
def residual_det_loss(anchor, positives, negatives, margin, dimensions=10):
    num_pos = positives.get_shape()[1]
    num_neg = negatives.get_shape()[1]

    pos_features = tf.subtract(positives, tf.tile(anchor, [1, int(num_pos), 1]))
    neg_features = tf.subtract(negatives, tf.tile(anchor, [1, int(num_neg), 1]))

    pos_s = tf.slice(tf.linalg.svd(pos_features, compute_uv=False), begin=[0, 0], size=[-1, dimensions])
    neg_s = tf.slice(tf.linalg.svd(neg_features, compute_uv=False), begin=[0, 0], size=[-1, dimensions])

    losses = tf.add(tf.subtract(tf.reduce_prod(pos_s, axis=1), tf.reduce_prod(neg_s, axis=1)), margin)
    return tf.reduce_mean(losses, axis=0)  # Mean over batches


# Modified triplet, use for hard positive mining
def evil_triplet_loss(q_vec, pos_vecs, neg_vecs, margin):
    worst_pos = worst_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]
    query_copies = tf.tile(q_vec, [1, int(num_neg), 1])
    worst_pos = tf.tile(tf.reshape(worst_pos, (-1, 1)), [1, int(num_neg)])
    m = tf.fill([int(batch), int(num_neg)], margin)
    triplet_loss = tf.reduce_mean(tf.reduce_sum(
        tf.maximum(tf.add(m, tf.subtract(worst_pos, tf.reduce_sum(tf.squared_difference(neg_vecs, query_copies), 2))),
                   tf.zeros([int(batch), int(num_neg)])), 1))
    return triplet_loss


# Modified quadrupled, use for hard positive mining
def evil_quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg, m1, m2):
    trip_loss = evil_triplet_loss(q_vec, pos_vecs, neg_vecs, m1)

    worst_pos = worst_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]

    other_neg_copies = tf.tile(other_neg, [1, int(num_neg), 1])
    worst_pos = tf.tile(tf.reshape(worst_pos, (-1, 1)), [1, int(num_neg)])
    m2 = tf.fill([int(batch), int(num_neg)], m2)

    second_loss = tf.reduce_mean(tf.reduce_sum(tf.maximum(
        tf.add(m2, tf.subtract(worst_pos, tf.reduce_sum(tf.squared_difference(neg_vecs, other_neg_copies), 2))),
        tf.zeros([int(batch), int(num_neg)])), 1))

    total_loss = trip_loss + second_loss

    return total_loss


def worst_pos_distance(query, pos_vecs):
    with tf.name_scope('best_pos_distance') as scope:
        num_pos = pos_vecs.get_shape()[1]
        query_copies = tf.tile(query, [1, int(num_pos), 1])  # shape num_pos x output_dim
        best_pos = tf.reduce_max(tf.reduce_sum(tf.squared_difference(pos_vecs, query_copies), 2), 1)
        return best_pos


def distance_loss(a_feature, pos_feature, squared_d_dists, d_max_squared, f_max_squared):
    scaled_d_dists, scaled_f_dists = _scale_distances(a_feature, pos_feature, squared_d_dists, d_max_squared,
                                                      f_max_squared)
    squared_diffs = tf.squared_difference(scaled_f_dists, scaled_d_dists)
    sum_of_squared_diffs = tf.reduce_mean(squared_diffs, 1)  # Mean over all positives
    return tf.reduce_mean(sum_of_squared_diffs, 0)  # Mean over batches


def huber_distance_loss(a_feature, pos_feature, squared_d_dists, d_max_squared, f_max_squared):
    scaled_d_dists, scaled_f_dists = _scale_distances(a_feature, pos_feature, squared_d_dists, d_max_squared,
                                                      f_max_squared)
    return tf.losses.huber_loss(scaled_d_dists, scaled_f_dists)


def distance_triplet_loss(a_feature, pos_features, neg_features, margin, lam, squared_d_dists, d_max_squared,
                          f_max_squared, triplet_loss_name='triplet_loss', distance_loss_name='huber_distance_loss'):
    """
    :param a_feature: Anchor
    :param pos_features: Positives
    :param neg_features: Negatives
    :param margin:
    :param lam: Scaling factor: loss = trip + lam*dist
    :param squared_d_dists: Squared distances from anchor to positives
    :param d_max_squared: Maximal squared distance
    :param f_max_squared: Maximal squared feature distance
    :param triplet_loss_name: triplet_loss or lazy_triplet_loss
    :param distance_loss_name: distance_loss or huber_distance_loss
    :return: loss
    """

    if 'huber' in distance_loss_name:
        return tf.add(getattr(pointnetvlad_cls, triplet_loss_name)
                      (a_feature, pos_features, neg_features, margin),
                      tf.multiply(lam, huber_distance_loss(a_feature, pos_features, squared_d_dists, d_max_squared,
                                                           f_max_squared)))
    else:
        return tf.add(getattr(pointnetvlad_cls, triplet_loss_name)
                      (a_feature, pos_features, neg_features, margin),
                      tf.multiply(lam, distance_loss(a_feature, pos_features, squared_d_dists, d_max_squared,
                                                     f_max_squared)))


def distance_quadruplet_loss(a_feature, pos_features, neg_features, other_neg, m1, m2, lam, squared_d_dists,
                             d_max_squared, f_max_squared, triplet_loss_name='triplet_loss',
                             distance_loss_name='huber_distance_loss'):
    """
    :param a_feature: Anchor
    :param pos_features: Positives
    :param neg_features: Negatives
    :param other_neg: Other negative
    :param m1:
    :param m2:
    :param lam: Scaling factor: loss = trip + lam*dist
    :param squared_d_dists: Squared distances from anchor to positives
    :param d_max_squared: Maximal squared distance
    :param f_max_squared: Maximal squared feature distance
    :param triplet_loss_name: triplet_loss or lazy_triplet_loss
    :param distance_loss_name: distance_loss or huber_distance_loss
    :return: loss
    """
    trip_loss = distance_triplet_loss(a_feature, pos_features, neg_features, m1, lam, squared_d_dists, d_max_squared,
                                      f_max_squared, triplet_loss_name, distance_loss_name)

    if 'huber' in distance_loss_name:
        best_pos = _best_huber_distance(a_feature, pos_features, squared_d_dists, d_max_squared, f_max_squared)
    else:
        best_pos = _best_distance(a_feature, pos_features, squared_d_dists, d_max_squared, f_max_squared)

    num_neg = neg_features.get_shape()[1]
    batch = a_feature.get_shape()[0]

    other_neg_copies = tf.tile(other_neg, [1, int(num_neg), 1])
    best_pos = tf.tile(tf.reshape(best_pos, (-1, 1)), [1, int(num_neg)])
    m2 = tf.fill([int(batch), int(num_neg)], m2)
    f_max_copies = tf.fill([int(batch), int(num_neg)], f_max_squared)

    second_loss = tf.reduce_mean(tf.reduce_max(tf.maximum(
        tf.add(m2, tf.subtract(best_pos, tf.div(tf.reduce_sum(tf.squared_difference(neg_features, other_neg_copies), 2),
                                                f_max_copies))),
        tf.zeros([int(batch), int(num_neg)])), 1))

    total_loss = trip_loss + second_loss
    return total_loss


# Helper functions
def _features2eigenvalues(features):
    f_len = features.get_shape()[2]
    all_features = tf.reshape(features, [-1, f_len])
    gram = tf.tensordot(all_features, tf.transpose(all_features), axes=1)
    eig, _ = tf.linalg.eigh(gram)
    return eig


def _pairwise_squared_distances(features):
    num_batches = features.get_shape()[0]
    r = tf.einsum('aij,aij->ai', features, features)
    r = tf.reshape(r, [num_batches, -1, 1])
    batch_product = tf.einsum('aij,ajk->aik', features, tf.transpose(features, perm=[0, 2, 1]))
    return r - 2 * batch_product + tf.transpose(r, perm=[0, 2, 1])


def _best_distance(a_feature, pos_features, squared_d_dists, d_max_squared, f_max_squared):
    scaled_d_dists, scaled_f_dists = _scale_distances(a_feature, pos_features, squared_d_dists, d_max_squared,
                                                      f_max_squared)
    squared_diffs = tf.squared_difference(scaled_f_dists, scaled_d_dists)
    return tf.reduce_min(squared_diffs, 1)


def _best_huber_distance(a_feature, pos_features, squared_d_dists, d_max_squared, f_max_squared):
    scaled_d_dists, scaled_f_dists = _scale_distances(a_feature, pos_features, squared_d_dists, d_max_squared,
                                                      f_max_squared)
    squared_diffs = tf.losses.huber_loss(scaled_f_dists, scaled_d_dists, reduction=tf.losses.Reduction.NONE)
    return tf.reduce_min(squared_diffs, 1)


def _scale_distances(a_feature, pos_feature, squared_d_dists, d_max_squared, f_max_squared):
    num_pos = pos_feature.get_shape()[1]
    batch_size = a_feature.get_shape()[0]
    a_feature_copies = tf.tile(a_feature, [1, int(num_pos), 1])  # shape num_pos x output_dim
    squared_f_dists = tf.reduce_sum(tf.squared_difference(pos_feature, a_feature_copies), 2)

    d_max_copies = tf.fill([int(batch_size), int(num_pos)], d_max_squared)
    f_max_copies = tf.fill([int(batch_size), int(num_pos)], f_max_squared)

    scaled_d_dists = tf.div(squared_d_dists, d_max_copies)
    scaled_f_dists = tf.div(squared_f_dists, f_max_copies)

    return scaled_d_dists, scaled_f_dists


def _min_eigenvalue(features):
    return tf.reduce_min(_features2eigenvalues(features))


def _max_eigenvalue(features):
    return tf.reduce_max(_features2eigenvalues(features))
