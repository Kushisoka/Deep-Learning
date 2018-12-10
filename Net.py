############################################################################
##
##	Name: Net.py						Author: Jose del Castillo Izquierdo
##
##	Description: Block & Sequence Constructor 
##
############################################################################


from tensorflow.keras import layers


###############################################################################################################
#
#		SEQUENCES (Net of Blocks)
#
###############################################################################################################


## If layers > 34

########################################################################################################################################################################
####### ResNet #######

def ResNetStage(input , size = 64, N = 3 , pooling = False ,data_format = None, kernel_regularizer = None, padding ='same', use_bias = True):

	for i in range(N):

		if i==0:			
			comun = ResNetBlockPooling(input, size, data_format, kernel_regularizer, padding, use_bias)
		else:
			comun = ResNetBlock(comun, size, data_format, kernel_regularizer, padding, use_bias)

	return comun

def FirstResNetStage(input , size = 64, N = 3 , pooling = False ,data_format = None, kernel_regularizer = None, padding ='same', use_bias = True):

	for i in range(N):

		if i==0:			
			comun = ResNetBlockPoolingFirst(input, size, data_format, kernel_regularizer, padding, use_bias)
		else:
			comun = ResNetBlock(comun, size, data_format, kernel_regularizer, padding, use_bias)

	return comun

####### END ResNet #######
########################################################################################################################################################################

########################################################################################################################################################################
####### PreActivated #######

def PreActivatedResNetStage(input , size = 64, N = 3 , pooling = False ,data_format = None, kernel_regularizer = None, padding ='same', use_bias = True):

	for i in range(N):

		if i==0:			
			comun = PreActivatedResNetBlockPooling(input, size, data_format, kernel_regularizer, padding, use_bias)
		else:
			comun = PreActivatedResNetBlock(comun, size, data_format, kernel_regularizer, padding, use_bias)

	return comun

def FirstPreActivatedResNetStage(input , size = 64, N = 3 , pooling = False ,data_format = None, kernel_regularizer = None, padding ='same', use_bias = True):

	for i in range(N):

		if i==0:			
			comun = PreActivatedResNetBlockPoolingFirst(input, size, data_format, kernel_regularizer, padding, use_bias)
		else:
			comun = PreActivatedResNetBlock(comun, size, data_format, kernel_regularizer, padding, use_bias)

	return comun

####### END PreActivated #######
########################################################################################################################################################################

########################################################################################################################################################################
####### ELU #######

def ELUResNetStage(input , size = 64, N = 3 , pooling = False ,data_format = None, kernel_regularizer = None, padding ='same', use_bias = True):

	for i in range(N):

		if i==0:			
			comun = ELUResNetBlockPooling(input, size, data_format, kernel_regularizer, padding, use_bias)
		else:
			comun = ELUResNetBlock(comun, size, data_format, kernel_regularizer, padding, use_bias)

	return comun

def FirstELUResNetStage(input , size = 64, N = 3 , pooling = False ,data_format = None, kernel_regularizer = None, padding ='same', use_bias = True):

	for i in range(N):

		if i==0:			
			comun = ELUResNetBlockPoolingFirst(input, size, data_format, kernel_regularizer, padding, use_bias)
		else:
			comun = ELUResNetBlock(comun, size, data_format, kernel_regularizer, padding, use_bias)

	return comun

####### END ResNet #######
########################################################################################################################################################################

########################################################################################################################################################################
####### Separable #######

def SeparableResNetStage(input , size = 64, N = 3 , pooling = False ,data_format = None, kernel_regularizer = None, padding ='same', use_bias = True):

	for i in range(N):

		if i==0:			
			comun = SeparableResNetBlockPooling(input, size, data_format, kernel_regularizer, padding, use_bias)
		else:
			comun = SeparableResNetBlock(comun, size, data_format, kernel_regularizer, padding, use_bias)

	return comun

def FirstSeparableResNetStage(input , size = 64, N = 3 , pooling = False ,data_format = None, kernel_regularizer = None, padding ='same', use_bias = True):

	for i in range(N):

		if i==0:			
			comun = SeparableResNetBlockPoolingFirst(input, size, data_format, kernel_regularizer, padding, use_bias)
		else:
			comun = SeparableResNetBlock(comun, size, data_format, kernel_regularizer, padding, use_bias)

	return comun

####### END Separable #######
########################################################################################################################################################################

## If layers < 34

def ResSequence(input , size = 256, N = 5, data_format = None, kernel_regularizer = None, padding ='same', use_bias = True):

	for i in range(N):

		if i==0:

			comun = ResBlock(input, size, data_format, kernel_regularizer, padding, use_bias)

		else:

			comun = ResBlock(comun, size, data_format, kernel_regularizer, padding, use_bias)

	return comun


###############################################################################################################
#
#		BLOCKS
#
###############################################################################################################

#####################
#					#
#	ResNetIntro		#
#					#
#####################

def ResNetIntro(input , size = 64, data_format = None, kernel_regularizer = None, padding ='same', use_bias = True):

	comun = layers.ZeroPadding2D(padding=(3, 3), data_format=data_format)(input)
	comun = layers.Conv2D(size, (7, 7), strides = 2, data_format=data_format , kernel_regularizer=kernel_regularizer, use_bias=use_bias)(comun)
	comun = layers.BatchNormalization()(comun)
	comun = layers.Activation("relu")(comun)
	comun = layers.ZeroPadding2D(padding=(1, 1), data_format=data_format)(comun)
	output = layers.MaxPooling2D(pool_size=(3, 3), strides = (2,2), data_format=data_format)(comun)

	print("Intro Output: " + str(output.shape))

	return output



#####################
#					#
#	Convolutional	#
#					#
#####################


##############################################################################################################################################################################################

####################################################################################################################################################################

#	Block V1:	ResNet

####################################################################################################################################################################

def ResBlock(input , size = 256, data_format = None, kernel_regularizer = None, padding ='same', use_bias = True):

	comun = layers.Conv2D(size, (3, 3), data_format=data_format , kernel_regularizer=kernel_regularizer, padding=padding, use_bias=use_bias)(input)
	comun = layers.BatchNormalization()(comun)
	comun = layers.Activation("relu")(comun)
	comun = layers.Conv2D(size, (3, 3), data_format=data_format , kernel_regularizer=kernel_regularizer, padding=padding, use_bias=use_bias)(comun)
	comun = layers.BatchNormalization()(comun)
	comun = layers.add([comun, input])
	output = layers.Activation("relu")(comun)

	return output


"""BLOCK V1 ++"""
def ResNetBlockPoolingFirst(input , size = 64, data_format = None, kernel_regularizer = None, padding ='same', use_bias = True):

	comun = layers.Conv2D(size, (1, 1), data_format=data_format , kernel_regularizer=kernel_regularizer, padding = padding, use_bias=use_bias)(input)
	comun = layers.BatchNormalization()(comun)
	comun = layers.Activation("relu")(comun)
	comun = layers.Conv2D(size, (3, 3), data_format=data_format , kernel_regularizer=kernel_regularizer, padding=padding, use_bias=use_bias)(comun)
	comun = layers.BatchNormalization()(comun)
	comun = layers.Activation("relu")(comun)
	comun = layers.Conv2D(size*4, (1, 1), data_format=data_format , kernel_regularizer=kernel_regularizer, padding=padding, use_bias=use_bias)(comun)
	comun = layers.BatchNormalization()(comun)

	res = layers.Conv2D(size*4, (1, 1), data_format=data_format , kernel_regularizer=kernel_regularizer, padding=padding, use_bias=use_bias)(input)
	res = layers.BatchNormalization()(res)

	comun = layers.add([comun, res])
	output = layers.Activation("relu")(comun)

	return output


"""BLOCK V1 ++"""
def ResNetBlock(input , size = 64, data_format = None, kernel_regularizer = None, padding ='same', use_bias = True):

	comun = layers.Conv2D(size, (1, 1), data_format=data_format , kernel_regularizer=kernel_regularizer, padding=padding, use_bias=use_bias)(input)
	comun = layers.BatchNormalization()(comun)
	comun = layers.Activation("relu")(comun)
	comun = layers.Conv2D(size, (3, 3), data_format=data_format , kernel_regularizer=kernel_regularizer, padding=padding, use_bias=use_bias)(comun)
	comun = layers.BatchNormalization()(comun)
	comun = layers.Activation("relu")(comun)
	comun = layers.Conv2D(size*4, (1, 1), data_format=data_format , kernel_regularizer=kernel_regularizer, padding=padding, use_bias=use_bias)(comun)
	comun = layers.BatchNormalization()(comun)

	comun = layers.add([comun, input])
	output = layers.Activation("relu")(comun)

	return output


"""BLOCK V1 ++"""
def ResNetBlockPooling(input , size = 64, data_format = None, kernel_regularizer = None, padding ='same', use_bias = True):

	print("ResNetBlockPooling Input: " + str(input.shape))

	comun = layers.Conv2D(size, (1, 1), strides=2, data_format=data_format , kernel_regularizer=kernel_regularizer, use_bias=use_bias)(input)
	comun = layers.BatchNormalization()(comun)
	comun = layers.Activation("relu")(comun)
	comun = layers.Conv2D(size, (3, 3), data_format=data_format , kernel_regularizer=kernel_regularizer, padding=padding, use_bias=use_bias)(comun)
	comun = layers.BatchNormalization()(comun)
	comun = layers.Activation("relu")(comun)
	comun = layers.Conv2D(size*4, (1, 1), data_format=data_format , kernel_regularizer=kernel_regularizer, padding=padding, use_bias=use_bias)(comun)
	comun = layers.BatchNormalization()(comun)

	res = layers.Conv2D(size*4, (1, 1), strides=2, data_format=data_format , kernel_regularizer=kernel_regularizer, use_bias=use_bias)(input)
	res = layers.BatchNormalization()(res)

	comun = layers.add([comun, res])
	output = layers.Activation("relu")(comun)

	return output

####################################################################################################################################################################

#	Block V1:	ResNet

####################################################################################################################################################################

##############################################################################################################################################################################################


##############################################################################################################################################################################################

####################################################################################################################################################################

#	Block V2:	PreActivatedResNet

####################################################################################################################################################################

"""BLOCK V2"""
def PreActivatedResBlock(input, size = 256 , data_format = None, kernel_regularizer = None, padding ='same', use_bias = True):

	comun = layers.BatchNormalization()(input)
	comun = layers.Activation("relu")(comun)
	comun = layers.Conv2D(size, (3, 3), data_format=data_format , kernel_regularizer=kernel_regularizer, padding=padding, use_bias=use_bias)(comun)
	comun = layers.BatchNormalization()(comun)
	comun = layers.Activation("relu")(comun)
	comun = layers.Conv2D(size, (3, 3), data_format=data_format , kernel_regularizer=kernel_regularizer, padding=padding, use_bias=use_bias)(comun)
	output = layers.add([comun, input])

	return output

"""BLOCK V2 ++"""
def PreActivatedResNetBlock(input, size = 64 , data_format = None, kernel_regularizer = None, padding ='same', use_bias = True):

	comun = layers.BatchNormalization()(input)
	comun = layers.Activation("relu")(comun)
	comun = layers.Conv2D(size, (1, 1), data_format=data_format , kernel_regularizer=kernel_regularizer, padding=padding, use_bias=use_bias)(comun)
	comun = layers.BatchNormalization()(comun)
	comun = layers.Activation("relu")(comun)
	comun = layers.Conv2D(size, (3, 3), data_format=data_format , kernel_regularizer=kernel_regularizer, padding=padding, use_bias=use_bias)(comun)
	comun = layers.BatchNormalization()(comun)
	comun = layers.Activation("relu")(comun)
	comun = layers.Conv2D(size*4, (1, 1), data_format=data_format , kernel_regularizer=kernel_regularizer, padding=padding, use_bias=use_bias)(comun)

	output = layers.add([comun, input])

	return output

"""BLOCK V2 ++"""
def PreActivatedResNetBlockPooling(input, size = 64 , data_format = None, kernel_regularizer = None, padding ='same', use_bias = True):

	print("PreActivatedResNetBlockPooling Input: " + str(input.shape))

	comun = layers.BatchNormalization()(input)
	comun = layers.Activation("relu")(comun)
	comun = layers.Conv2D(size, (1, 1), strides=2, data_format=data_format , kernel_regularizer=kernel_regularizer, use_bias=use_bias)(comun)
	comun = layers.BatchNormalization()(comun)
	comun = layers.Activation("relu")(comun)
	comun = layers.Conv2D(size, (3, 3), data_format=data_format , kernel_regularizer=kernel_regularizer, padding=padding, use_bias=use_bias)(comun)
	comun = layers.BatchNormalization()(comun)
	comun = layers.Activation("relu")(comun)
	comun = layers.Conv2D(size*4, (1, 1), data_format=data_format , kernel_regularizer=kernel_regularizer, padding=padding, use_bias=use_bias)(comun)

	res = layers.BatchNormalization()(input)
	res = layers.Activation("relu")(res)
	res = layers.Conv2D(size*4, (1, 1), strides=2, data_format=data_format , kernel_regularizer=kernel_regularizer, use_bias=use_bias)(res)
	
	output = layers.add([comun, res])


	return output

"""BLOCK V2 ++"""
def PreActivatedResNetBlockPoolingFirst(input, size = 64 , data_format = None, kernel_regularizer = None, padding ='same', use_bias = True):

	print("PreActivatedResNetBlockPooling Input: " + str(input.shape))

	comun = layers.BatchNormalization()(input)
	comun = layers.Activation("relu")(comun)
	comun = layers.Conv2D(size, (1, 1), data_format=data_format , kernel_regularizer=kernel_regularizer, use_bias=use_bias)(comun)
	comun = layers.BatchNormalization()(comun)
	comun = layers.Activation("relu")(comun)
	comun = layers.Conv2D(size, (3, 3), data_format=data_format , kernel_regularizer=kernel_regularizer, padding=padding, use_bias=use_bias)(comun)
	comun = layers.BatchNormalization()(comun)
	comun = layers.Activation("relu")(comun)
	comun = layers.Conv2D(size*4, (1, 1), data_format=data_format , kernel_regularizer=kernel_regularizer, padding=padding, use_bias=use_bias)(comun)

	res = layers.BatchNormalization()(input)
	res = layers.Activation("relu")(res)
	res = layers.Conv2D(size*4, (1, 1), data_format=data_format , kernel_regularizer=kernel_regularizer, use_bias=use_bias)(res)
	
	output = layers.add([comun, res])


	return output

####################################################################################################################################################################

#	Block V2:	PreActivatedResNet

####################################################################################################################################################################

##############################################################################################################################################################################################

##############################################################################################################################################################################################

####################################################################################################################################################################

#	Block V3:	ELUResNet

####################################################################################################################################################################

"""BLOCK V3"""
def EluResBlock(input, size = 256 , data_format = None, kernel_regularizer = None, padding ='same', use_bias = True):

	comun = layers.Conv2D(size, (3, 3), data_format=data_format , kernel_regularizer=kernel_regularizer, padding=padding, use_bias=use_bias)(input)
	comun = layers.Activation("elu")(comun)
	comun = layers.Conv2D(size, (3, 3), data_format=data_format , kernel_regularizer=kernel_regularizer, padding=padding, use_bias=use_bias)(comun)
	comun = layers.BatchNormalization()(comun)
	output = layers.add([comun, input])

	return output

####################################################################################################################################################################

#	Block V3:	ELUResNet

####################################################################################################################################################################

##############################################################################################################################################################################################

##############################################################################################################################################################################################

####################################################################################################################################################################

#	Block V4:	SeparableResNet

####################################################################################################################################################################

"""BLOCK V4 ++"""
def SeparableResNetBlockPoolingFirst(input , size = 64, data_format = None, kernel_regularizer = None, padding ='same', use_bias = True):

	comun = layers.SeparableConv2D(size, (1, 1), data_format=data_format , kernel_regularizer=kernel_regularizer, padding = padding, use_bias=use_bias)(input)
	comun = layers.BatchNormalization()(comun)
	comun = layers.Activation("relu")(comun)
	comun = layers.SeparableConv2D(size, (3, 3), data_format=data_format , kernel_regularizer=kernel_regularizer, padding=padding, use_bias=use_bias)(comun)
	comun = layers.BatchNormalization()(comun)
	comun = layers.Activation("relu")(comun)
	comun = layers.SeparableConv2D(size*4, (1, 1), data_format=data_format , kernel_regularizer=kernel_regularizer, padding=padding, use_bias=use_bias)(comun)
	comun = layers.BatchNormalization()(comun)

	res = layers.SeparableConv2D(size*4, (1, 1), data_format=data_format , kernel_regularizer=kernel_regularizer, padding=padding, use_bias=use_bias)(input)
	res = layers.BatchNormalization()(res)

	comun = layers.add([comun, res])
	output = layers.Activation("relu")(comun)

	return output


"""BLOCK V4 ++"""
def SeparableResNetBlock(input , size = 64, data_format = None, kernel_regularizer = None, padding ='same', use_bias = True):

	comun = layers.SeparableConv2D(size, (1, 1), data_format=data_format , kernel_regularizer=kernel_regularizer, padding=padding, use_bias=use_bias)(input)
	comun = layers.BatchNormalization()(comun)
	comun = layers.Activation("relu")(comun)
	comun = layers.SeparableConv2D(size, (3, 3), data_format=data_format , kernel_regularizer=kernel_regularizer, padding=padding, use_bias=use_bias)(comun)
	comun = layers.BatchNormalization()(comun)
	comun = layers.Activation("relu")(comun)
	comun = layers.SeparableConv2D(size*4, (1, 1), data_format=data_format , kernel_regularizer=kernel_regularizer, padding=padding, use_bias=use_bias)(comun)
	comun = layers.BatchNormalization()(comun)

	comun = layers.add([comun, input])
	output = layers.Activation("relu")(comun)

	return output


"""BLOCK V4 ++"""
def SeparableResNetBlockPooling(input , size = 64, data_format = None, kernel_regularizer = None, padding ='same', use_bias = True):

	print("ResNetBlockPooling Input: " + str(input.shape))

	comun = layers.SeparableConv2D(size, (1, 1), strides=2, data_format=data_format , kernel_regularizer=kernel_regularizer, use_bias=use_bias)(input)
	comun = layers.BatchNormalization()(comun)
	comun = layers.Activation("relu")(comun)
	comun = layers.SeparableConv2D(size, (3, 3), data_format=data_format , kernel_regularizer=kernel_regularizer, padding=padding, use_bias=use_bias)(comun)
	comun = layers.BatchNormalization()(comun)
	comun = layers.Activation("relu")(comun)
	comun = layers.SeparableConv2D(size*4, (1, 1), data_format=data_format , kernel_regularizer=kernel_regularizer, padding=padding, use_bias=use_bias)(comun)
	comun = layers.BatchNormalization()(comun)

	res = layers.SeparableConv2D(size*4, (1, 1), strides=2, data_format=data_format , kernel_regularizer=kernel_regularizer, use_bias=use_bias)(input)
	res = layers.BatchNormalization()(res)

	comun = layers.add([comun, res])
	output = layers.Activation("relu")(comun)

	return output

"""BLOCK V4"""
def SeparableResBlock(input, size = 256 , data_format = None, kernel_regularizer = None, padding ='same', use_bias = True):

	comun = layers.SeparableConv2D(size, (3, 3), data_format=data_format , kernel_regularizer=kernel_regularizer, padding=padding, use_bias=use_bias)(input)
	comun = layers.BatchNormalization()(comun)
	comun = layers.Activation("relu")(comun)
	comun = layers.SeparableConv2D(size, (3, 3), data_format=data_format , kernel_regularizer=kernel_regularizer, padding=padding, use_bias=use_bias)(comun)
	comun = layers.BatchNormalization()(comun)
	comun = layers.add([comun, input])
	output = layers.Activation("relu")(comun)

	return output

####################################################################################################################################################################

#	Block V4:	SeparableResNet

####################################################################################################################################################################

##############################################################################################################################################################################################

##############################################################################################################################################################################################

####################################################################################################################################################################

#	Block V5:	NASNet

####################################################################################################################################################################

"""BLOCK V5 (WIP)""" 
def NASBlock(input, size = 256 , data_format = None, kernel_regularizer = None, padding ='same', use_bias = True):

	comun = layers.SeparableConv2D(size, (3, 3), data_format=data_format , kernel_regularizer=kernel_regularizer, padding=padding, use_bias=use_bias)(input)
	comun = layers.BatchNormalization()(comun)
	comun = layers.Activation("relu")(comun)
	comun = layers.SeparableConv2D(size, (3, 3), data_format=data_format , kernel_regularizer=kernel_regularizer, padding=padding, use_bias=use_bias)(comun)
	comun = layers.BatchNormalization()(comun)
	comun = layers.add([comun, input])
	output = layers.Activation("relu")(comun)

	return output

####################################################################################################################################################################

#	Block V5:	NASNet

####################################################################################################################################################################

##############################################################################################################################################################################################

##############################################################################################################################################################################################

####################################################################################################################################################################

#	Block V6:	SEResNet

####################################################################################################################################################################

"""BLOCK V6""" 
def SEResBlock(input, size = 256, r = 4, data_format = None, kernel_regularizer = None, padding ='same', use_bias = True):

	u = ResBlock(input, size = size, data_format = data_format, kernel_regularizer = kernel_regularizer, padding =padding, use_bias = use_bias)

	s = layers.GlobalAveragePooling2D(data_format=data_format)(u)
	s = layers.Dense(size/r, kernel_regularizer=kernel_regularizer, use_bias=use_bias)(s)
	s = layers.Activation("relu")(s)
	s = layers.Dense(size, kernel_regularizer=kernel_regularizer, use_bias=use_bias)(s)
	s = layers.Activation("sigmoid")(s)

	scale = layers.Multiply([u, s])
	output = layers.add([scale, input])

	return output


####################################################################################################################################################################

#	Block V6:	SEResNet

####################################################################################################################################################################

##############################################################################################################################################################################################




#######################################################################
##
##		EXPERIMENTAL
##
#######################################################################

"""BLOCK V3(V4)"""
def EluSeparableResBlock(input, size = 256 , data_format = None, kernel_regularizer = None, padding ='same', use_bias = True):

	comun = layers.SeparableConv2D(size, (3, 3), data_format=data_format , kernel_regularizer=kernel_regularizer, padding=padding, use_bias=use_bias)(input)
	comun = layers.Activation("elu")(comun)
	comun = layers.SeparableConv2D(size, (3, 3), data_format=data_format , kernel_regularizer=kernel_regularizer, padding=padding, use_bias=use_bias)(comun)
	comun = layers.BatchNormalization()(comun)
	output = layers.add([comun, input])

	return output

