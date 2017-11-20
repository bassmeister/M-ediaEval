##---------------------------------------------
## Cedric Bezy
## 20 nov.
##
## mediaeval
##---------------------------------------------


##=====================================================================
## Packages
##=====================================================================

library(dplyr)
library(FactoMineR)
library(tibble)

library(ggplot2)

source('~/PROJETS/R/MainProject/functions/gf_plots.R')
source('~/PROJETS/R/MainProject/functions/gf_settings.R')
source('~/PROJETS/R/MainProject/functions/gf_analysis.R')


##=====================================================================
## Dict categ
##=====================================================================

target_categs = c('1001' = 'Autos and vehicule',
                  '1009' = 'Food and drink',
                  '1011' = 'Health',
                  '1013' = 'Movies and television',
                  '1014' = 'Litterature',
                  '1016' = 'Politics',
                  '1017' = 'Religion',
                  '1019' = 'Sports')

##=====================================================================
## Import Data
##=====================================================================

countsTags_categs <- read.csv2("output/count_tags_categs.csv",
                               stringsAsFactors = FALSE,
                               allowEscapes = TRUE,
                               row.names = "tag")

##-----------------------------------------
## ACP
##-----------------------------------------

acp_cat <- FactoMineR::PCA(countsTags_categs,
                       scale.unit = FALSE)

gf_plotInertia(acp_cat, choix = "cum")

## Individus
coords_categs <- as.data.frame(acp_cat$ind$coord)

## Kmeans
km <- list(cluster = 0)
while(min(table(km$cluster)) <= 2){
    ## kmeans
    km <- kmeans(
        coords_categs,
        centers = length(target_categs),
        iter.max = 20
    )
    print(table(km$cluster))
}


coords_categs$km_clust <- as.factor(km$cluster)

##-----------------------------------------
## Plots
##-----------------------------------------

pltCategs <- ggplot(mapping = aes(x = Dim.1, y = Dim.2, color = km_clust),
                  data = coords_categs) +
    geom_point(
        size = 3
    ) + 
    stat_ellipse(type = "euclid", level = 7) + 
    theme_bw(base_size = 13) +
    ggtitle("PCA of tags for each category",
            "with a kmeans clustering coloration.")
pltCategs

plot.PCA(acp_cat,
         choix = "var")



##=====================================================================
## Import Data
##=====================================================================

countsTags_docs <- read.csv2("output/count_tags_docs.csv",
                             stringsAsFactors = FALSE,
                             allowEscapes = TRUE,
                             row.names = "iddoc",
                             dec = ".")

##-----------------------------------------
## ACP
##-----------------------------------------

acp_docs <- FactoMineR::PCA(countsTags_docs,
                            scale.unit = FALSE,
                            quali.sup = 1)

## Individus
coords_docs <- as.data.frame(acp_docs$ind$coord)

## Kmeans
km <- list(cluster = 0)
while(min(table(km$cluster)) <= 2){
    km <- kmeans(
        coords_docs,
        centers = length(target_categs),
        iter.max = 20
    )
    print(table(km$cluster))
}

coords_docs$clust <- as.factor(km$cluster)
coords_docs$categ <- countsTags_docs$categ


with(coords_docs, {
    table(clust, categ)
})



