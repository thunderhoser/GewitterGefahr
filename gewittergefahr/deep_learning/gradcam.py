"""Methods for Grad-CAM (gradient-weighted class-activation-mapping).

Most of this was scavenged from:
https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py

--- REFERENCE ---

Selvaraju, R.R., M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra,
2017: "Grad-CAM: Visual explanations from deep networks via gradient-based
localization".  International Conference on Computer Vision, IEEE,
https://doi.org/10.1109/ICCV.2017.74.
"""

import numpy
import keras
from keras import backend as K
from keras.applications.vgg16 import VGG16
import tensorflow
from tensorflow.python.framework import ops as tensorflow_ops
from cv2 import resize as cv2_resize
from gewittergefahr.gg_utils import error_checking

SMALL_NUMBER = 1e-5


def _register_gradient():
    """Registers gradient with TensorFlow backend?

    :return: Don't know yet.
    """

    if 'GuidedBackProp' not in tensorflow_ops._gradient_registry._registry:
        @tensorflow_ops.RegisterGradient('GuidedBackProp')
        def _GuidedBackProp(operation, gradient_tensor):
            input_type = operation.inputs[0].dtype

            return (
                gradient_tensor *
                tensorflow.cast(gradient_tensor > 0., input_type) *
                tensorflow.cast(operation.inputs[0] > 0., input_type)
            )


def _normalize_tensor(input_tensor):
    """Normalizes tensor by its L2 norm.

    :param input_tensor: Unnormalized tensor.
    :return: output_tensor: Normalized tensor.
    """

    rms_tensor = K.sqrt(K.mean(K.square(input_tensor)))
    return input_tensor / (rms_tensor + SMALL_NUMBER)


def _compute_gradients(loss_tensor, list_of_input_tensors):
    """Computes gradient of each input tensor with respect to loss tensor.

    :param loss_tensor: Loss tensor.
    :param list_of_input_tensors: 1-D list of input tensors.
    :return: list_of_gradient_tensors: 1-D list of gradient tensors.
    """

    list_of_gradient_tensors = tensorflow.gradients(
        loss_tensor, list_of_input_tensors)

    for i in range(len(list_of_gradient_tensors)):
        if list_of_gradient_tensors[i] is not None:
            continue

        list_of_gradient_tensors[i] = tensorflow.zeros_like(
            list_of_input_tensors[i])

    return list_of_gradient_tensors


def _change_backprop_function(model_object,
                              backprop_function_name='GuidedBackProp'):
    """Changes backpropagation function for Keras model.

    :param model_object: Instance of `keras.models.Model` or
        `keras.models.Sequential`.
    :param backprop_function_name: Name of backprop function that will replace
        the default.
    :return: new_model_object: Same as `model_object` but with new backprop
        function.
    """

    # TODO(thunderhoser): I don't think this method even works.

    graph_object = tensorflow.get_default_graph()

    with graph_object.gradient_override_map({'Relu': backprop_function_name}):
        activation_layer_dict = dict([
            (lyr.name, lyr) for lyr in model_object.layers[1:]
            if hasattr(lyr, 'activation')
        ])

        for this_key in activation_layer_dict:

            # TODO(thunderhoser): What about layers with other activation
            # functions?
            if (activation_layer_dict[this_key].activation ==
                    keras.activations.relu):
                activation_layer_dict[this_key].activation = tensorflow.nn.relu

        new_model_object = VGG16(weights='imagenet')
        new_model_object.summary()

    list_of_activn_layer_objects = [
        lyr for lyr in new_model_object.layers[1:] if hasattr(lyr, 'activation')
    ]

    for this_layer_object in list_of_activn_layer_objects:
        print this_layer_object.activation

    return new_model_object


def _make_saliency_function(model_object, layer_name):
    """Creates saliency function.

    :param model_object: Instance of `keras.models.Model` or
        `keras.models.Sequential`.
    :param layer_name: Saliency will be computed with respect to activations in
        this layer.
    :return: saliency_function: Instance of `keras.backend.function`.
    """

    layer_dict = dict([
        (lyr.name, lyr) for lyr in model_object.layers[1:]
    ])

    # TODO(thunderhoser): Do I need list of input tensors?
    output_tensor = layer_dict[layer_name].output
    max_output_tensor = K.max(output_tensor, axis=-1)
    saliency_tensor = K.gradients(
        K.sum(max_output_tensor), model_object.input
    )[0]

    return K.function(
        [model_object.input, K.learning_phase()], [saliency_tensor]
    )


def _normalize_guided_gradcam_output(image_matrix):
    """Normalizes image produced by guided Grad-CAM.

    M = number of rows in grid
    N = number of columns in grid
    C = number of channels

    :param image_matrix: numpy array with output of guided Grad-CAM.  Dimensions
        may be 1 x M x N x C or M x N x C.
    :return: image_matrix: Same as input, except that values are normalized and
        dimensions are always M x N x C.
    """

    if image_matrix.shape[0] == 1:
        image_matrix = image_matrix[0, ...]

    # Standardize.
    image_matrix -= numpy.mean(image_matrix)
    image_matrix /= (numpy.std(image_matrix, ddof=0) + SMALL_NUMBER)

    # Force standard deviation of 0.1 and mean of 0.5.
    image_matrix = 0.5 + image_matrix * 0.1
    image_matrix[image_matrix < 0.] = 0.
    image_matrix[image_matrix > 1.] = 1.

    return image_matrix


def run_gradcam(model_object, input_matrix, target_class, conv_layer_name):
    """Runs Grad-CAM.

    M = number of rows in grid
    N = number of columns in grid

    :param model_object: Trained instance of `keras.models.Model` or
        `keras.models.Sequential`.  Grad-CAM will be run for this model.
    :param input_matrix: numpy array containing one input example.  Array
        dimensions must be the same as input dimensions for `model_object`.
    :param target_class: Target class (integer from 0...[K - 1], where K =
        number of classes).  The class-activation map (CAM) will be created for
        this class.
    :param conv_layer_name: Name of convolutional layer.  Neuron-importance
        weights will be based on activations in this layer.
    :return: class_activation_matrix: M-by-N numpy array of class activations.
    """

    # TODO(thunderhoser): Check input dimensions.

    num_output_neurons = model_object.layers[-1].output.get_shape().as_list()[
        -1]

    if num_output_neurons == 1:
        error_checking.assert_is_leq(target_class, 1)
        if target_class == 1:
            loss_tensor = model_object.layers[-1].output[..., 0]
        else:
            loss_tensor = 1 - model_object.layers[-1].output[..., 0]
    else:
        error_checking.assert_is_less_than(target_class, num_output_neurons)
        loss_tensor = model_object.layers[-1].output[..., target_class]

    # TODO(thunderhoser): Need post-activation values.
    conv_layer_activation_tensor = model_object.get_layer(
        name=conv_layer_name
    ).output
    gradient_tensor = _compute_gradients(
        loss_tensor, [conv_layer_activation_tensor]
    )[0]
    gradient_tensor = _normalize_tensor(gradient_tensor)

    gradient_function = K.function(
        [model_object.input],
        [conv_layer_activation_tensor, gradient_tensor]
    )

    conv_layer_activation_matrix, gradient_matrix = gradient_function(
        [input_matrix])
    conv_layer_activation_matrix = conv_layer_activation_matrix[0, ...]
    gradient_matrix = gradient_matrix[0, ...]

    weight_by_conv_filter = numpy.mean(gradient_matrix, axis=(0, 1))
    class_activation_matrix = numpy.ones(conv_layer_activation_matrix.shape[:2])

    num_conv_filters = len(weight_by_conv_filter)
    for m in range(num_conv_filters):
        class_activation_matrix += (
            weight_by_conv_filter[m] * conv_layer_activation_matrix[..., m]
        )

    num_input_rows = input_matrix.shape[1]
    num_input_columns = input_matrix.shape[2]
    class_activation_matrix = cv2_resize(
        class_activation_matrix, (num_input_rows, num_input_columns)
    )

    class_activation_matrix[class_activation_matrix < 0.] = 0.
    return class_activation_matrix / numpy.max(class_activation_matrix)


def run_guided_gradcam(model_object, input_matrix, conv_layer_name,
                       class_activation_matrix):
    """Runs guided Grad-CAM.

    M = number of rows in grid
    N = number of columns in grid
    C = number of channels

    :param model_object: See doc for `run_gradcam`.
    :param input_matrix: Same.
    :param conv_layer_name: Same.
    :param class_activation_matrix: Matrix created by `run_gradcam`.
    :return: gradient_matrix: M-by-N-by-C numpy array of gradients.
    """

    _register_gradient()

    new_model_object = _change_backprop_function(model_object=model_object)
    saliency_function = _make_saliency_function(
        model_object=new_model_object, layer_name=conv_layer_name)

    saliency_matrix = saliency_function([input_matrix, 0])[0]
    gradient_matrix = saliency_matrix * class_activation_matrix[
        ..., numpy.newaxis]
    return _normalize_guided_gradcam_output(gradient_matrix)
