library(tidyverse)
library(ggpubr)
library(ggplot2)

# data sets ####
unsorted <- read_csv("breast-cancer.csv")
ids <- unsorted[,1]
diagnosis <- unsorted[,2]
radius <- data.frame(unsorted[,3],unsorted[,13],unsorted[,23])
names(radius) <- c("mean", "SD", "Worst")
texture <- data.frame(unsorted[,4],unsorted[,14],unsorted[,24])
names(texture) <- c("mean", "SD", "Worst")
perimeter <- data.frame(unsorted[,5],unsorted[,15],unsorted[,25])
names(perimeter) <- c("mean", "SD", "Worst")
area <- data.frame(unsorted[,6],unsorted[,16],unsorted[,26])
names(area) <- c("mean", "SD", "Worst")
smoothness <- data.frame(unsorted[,7],unsorted[,17],unsorted[,27])
names(smoothness) <- c("mean", "SD", "Worst")
compactness <- data.frame(unsorted[,8],unsorted[,18],unsorted[,28])
names(compactness) <- c("mean", "SD", "Worst")
concavity <- data.frame(unsorted[,9],unsorted[,19],unsorted[,29])
names(concavity) <- c("mean", "SD", "Worst")
concave_points <- data.frame(unsorted[,10],unsorted[,20],unsorted[,30])
names(concave_points) <- c("mean", "SD", "Worst")
symmetry <- data.frame(unsorted[,11],unsorted[,21],unsorted[,31])
names(symmetry) <- c("mean", "SD", "Worst")
fractal <- data.frame(unsorted[,12],unsorted[,22],unsorted[,32])
names(fractal) <- c("mean", "SD", "Worst")

# properties visualization ####

rad_plot <- ggplot(data = radius, aes(x=mean)) +
  geom_density()+
  labs(x="",y="", title = "Radius")+
  theme_classic()

per_plot <- ggplot(data = perimeter, aes(x=mean)) +
  geom_density()+
  labs(x="",y="", title = "Perimeter")+
  theme_classic()

smooth_plot <- ggplot(data = smoothness, aes(x=mean)) +
  geom_density()+
  labs(x="",y="", title = "Smoothness")+
  theme_classic()

tex_plot <- ggplot(data = texture, aes(x=mean)) +
  geom_density()+
  labs(x="",y="", title = "Texture")+
  theme_classic()

sym_plot <- ggplot(data = symmetry, aes(x=mean)) +
  geom_density()+
  labs(x="",y="", title = "Symmetries")+
  theme_classic()

frac_plot <- ggplot(data = fractal, aes(x=mean)) +
  geom_density()+
  labs(x="",y="", title = "Fractal Dimensions")+
  theme_classic()

con_plot <- ggplot(data = concavity, aes(x=mean)) +
  geom_density()+
  labs(x="",y="", title = "Concavity")+
  theme_classic()

points_plot <- ggplot(data = concave_points, aes(x=mean)) +
  geom_density()+
  labs(x="",y="", title = "Concave Points")+
  theme_classic()

comp_plot <- ggplot(data = compactness, aes(x=mean)) +
  geom_density()+
  labs(x="",y="", title = "Compactness")+
  theme_classic()

all_plots <- ggarrange(comp_plot, con_plot, per_plot, rad_plot, smooth_plot,
                       tex_plot, points_plot, frac_plot, sym_plot,
                       labels = c("A", "B", "C", "D", "E", "F", "G", "H", "I", "J"),
                       ncol = 3, nrow = 3)
all_plots

# compared to malignant/benign visualization ####
radius$diagnosis <- as.factor(unlist(diagnosis))
rad_box <- ggplot(data = radius, aes(x = diagnosis,
                          y = mean))+ 
  geom_boxplot()+
  labs(x = "Benign or Malignant",
       y = "Mean Radius")+
  theme_classic()

mal_rad <- radius$mean[radius$diagnosis == "M"]
ben_rad <- radius$mean[radius$diagnosis == "B"]