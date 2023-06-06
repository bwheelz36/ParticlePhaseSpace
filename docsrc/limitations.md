# Limitations

The major limitation of this code at the time of writing is that it can only easily handle data that fits inside memory. This is partially a result of the choice of pandas as the backend - [this page](https://pandas.pydata.org/docs/user_guide/scale.html) describes some difficulties and solutions of handling large data with pandas.

As is discussed in the above link: if your data is too big to fit inside RAM, it should be possible to read and process your data in 'chunks' where each chunk can fit inside memory. This is not supported in most data loaders, but should be possible with minimal extensions - open an issue and we can talk about it! 

Beyond this, libraries such as [DASK](https://www.dask.org/) may enable using this library of distributed resources. This is discussed a little bit in [this issue](https://github.com/bwheelz36/ParticlePhaseSpace/issues/158), with an example of utilising DASK on an OpenPMD dataset. 