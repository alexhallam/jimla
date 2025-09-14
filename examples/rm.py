import polars as pl
import numpy as np
from jimla import lm, augment, glance

mtcars_path = 'https://gist.githubusercontent.com/seankross/a412dfbd88b3db70b74b/raw/5f23f993cd87c283ce766e7ac6b329ee7cc2e1d1/mtcars.csv'
df = pl.read_csv(mtcars_path)

# lm() automatically prints tidy output
result = lm(df, "mpg ~ cyl + wt*hp - 1")

# Show augmented data
augment(result, df)

# Show model summary
glance(result)