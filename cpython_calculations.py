import ctypes
# The command below assumes you are using windows
clibrary = ctypes.WinDLL("./cpython.dll")
# if on mac os, change the above line to the following:
# clibrary = ctypes.CDLL("./cpython.so")



predict_weight = clibrary.predict_weight_vector
predict_weight.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
predict_weight.restype = ctypes.POINTER(ctypes.c_double)


free_func = clibrary.free_memory
free_func.argtypes = [ctypes.POINTER(ctypes.c_double)]


predict_emotion = clibrary.get_probability
predict_emotion.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int32, ctypes.c_int32]
predict_emotion.restype = ctypes.c_double



def get_new_weight_vec(tokens, weights, emotion_int, size = 500, num_of_emotions = 11):
    """
        tokens is the tokenenized + embedded input vector
        weights is the dictionary containing the weights of each emotion
        emotion is the actual emotion value
        size is the side of the tokens and weights vectors

        the function outputs a new weight vector 
    """


    c_tokens = (ctypes.c_double * (size))(*tokens, 0)
    c_weights = (ctypes.c_double * (size))()
    c_size = ctypes.c_int32(size)
    c_token_length = ctypes.c_int32(len(tokens) + 1)

        
    


    # size will be equal to 500 on passes (for now at least)
    for i in range(num_of_emotions):
        y = 0
        weight = weights[str(i)]
        # add bias scalar too calculation

        for j in range(size):
            c_weights[j] = weight[j]

        if i == emotion_int:
            y = ctypes.c_int32(1)
        else:
            y = ctypes.c_int32(0)
        
        new_weight_vector = predict_weight(c_tokens, c_weights, c_token_length, c_size, y, ctypes.c_int32(i))


        for j in range(size):
            weight[j] = float(new_weight_vector[j])
        
        
        free_func(new_weight_vector)
        weights[str(i)] = weight



    return weights




def get_predicted_emotions(tokens, weights, size = 500, num_of_emotions = 11):
    """
        tokens is the tokenenized + embedded input vector
        weights is the dictionary containing the weights of each emotion
        emotion is the actual emotion value
        size is the side of the tokens and weights vectors

        the function outputs a new weight vector 
    """


    c_tokens = (ctypes.c_double * (size))(*tokens, 0)
    c_weights = (ctypes.c_double * (size))()
    c_size = ctypes.c_int32(size)
    c_token_length = ctypes.c_int32(len(tokens) + 1)

    # size will be equal to 500 on passes (for now at least)

    list_of_probabilites = [
                    ["Happy",0],
                    ["Sad",0],
                    ["Angry",0],
                    ["Confused",0],
                    ["Frustrated",0],
                    ["Scared",0],
                    ["Suprised",0],
                    ["Disgusted",0],
                    ["Anxious",0],
                    ["Shame",0],
                    ["Excited",0]
                ]
    for i in range(num_of_emotions):
        y = 0
        weight = weights[str(i)]
        # add bias scalar too calculation

        # second loop for individual weight vectors
        for j in range(size):
            c_weights[j] = weight[j]

        # calculate probability for each individual emotion
        probability = predict_emotion(c_tokens, c_weights, c_token_length, c_size)

        list_of_probabilites[i][1] += round(float(probability), 6)


    return list_of_probabilites