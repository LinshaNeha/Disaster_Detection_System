install.packages("keras")
install.packages("tensorflow")
install.packages("reticulate")
install.packages("magick")

library(keras)
library(tensorflow)
library(reticulate)
library(magick)
tf$constant("Hello from TensorFlow")
resize_images <- function(input_folder, output_folder, size = "128x128") {
  if (!dir.exists(output_folder)) dir.create(output_folder, recursive = TRUE)
  
  files <- list.files(input_folder, full.names = TRUE)
  
  for (file in files) {
    img <- image_read(file)
    img_resized <- image_resize(img, size)
    out_file <- file.path(output_folder, basename(file))
    image_write(img_resized, out_file)
  }
}

# Resize all folders (once)
resize_images("C:/Users/linsha neha/OneDrive/Documents/Desktop/dataset creation/Normal", "C:/Users/linsha neha/OneDrive/Documents/Desktop/dataset creation/resized/normal")
resize_images("C:/Users/linsha neha/OneDrive/Documents/Desktop/dataset creation/Flood", "C:/Users/linsha neha/OneDrive/Documents/Desktop/dataset creation/resized/flood")
resize_images("C:/Users/linsha neha/OneDrive/Documents/Desktop/dataset creation/Wildfire", "C:/Users/linsha neha/OneDrive/Documents/Desktop/dataset creation/resized/wildfire")
resize_images("C:/Users/linsha neha/OneDrive/Documents/Desktop/dataset creation/Drought", "C:/Users/linsha neha/OneDrive/Documents/Desktop/dataset creation/resized/drought")
img_width <- 128
img_height <- 128
batch_size <- 32

train_data_dir <- "C:/Users/linsha neha/OneDrive/Documents/Desktop/dataset creation/resized"
datagen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

train_generator <- flow_images_from_directory(
  train_data_dir,
  target_size = c(img_width, img_height),
  batch_size = batch_size,
  class_mode = "categorical",
  subset = "training",
  seed = 123,
  generator = datagen
)

validation_generator <- flow_images_from_directory(
  train_data_dir,
  target_size = c(img_width, img_height),
  batch_size = batch_size,
  class_mode = "categorical",
  subset = "validation",
  seed = 123,
  generator = datagen
)
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu', input_shape = c(img_width, img_height, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = length(train_generator$class_indices), activation = 'softmax')

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = 'accuracy'
)
history <- model %>% fit(
  train_generator,
  steps_per_epoch = ceiling(train_generator$samples / batch_size),
  epochs = 10,
  validation_data = validation_generator,
  validation_steps = ceiling(validation_generator$samples / batch_size)
)

plot(history)
model %>% save_model_hdf5("C:/Users/linsha neha/OneDrive/Documents/Desktop/dataset creation/model/disaster_classifier_model.h5")



