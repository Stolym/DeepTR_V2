$	?????rf@s???os@hY??????!ir1?Հ@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'ir1?Հ@?o&???@1ѯ?_?~@I?
E??2@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails 4??s??0?????1? 3??O\?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!hY??????1???מY??I?mO???~?r11*	33333Cx@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapp_?Q??!Kqќ/{J@)z?):????1?\@???G@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?b?=y??!?3Ua?8@)?o_???1?<d?&51@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?1w-!??!ա!?N?$@)Έ?????1?}?AV*#@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat??H?}??!??ç??@)F%u???1L?sD?3@:Preprocessing2T
Iterator::Root::ParallelMapV2??_?L??!D?b?n@)??_?L??1D?b?n@:Preprocessing2E
Iterator::Rootsh??|???!????wa%@)Q?|a2??1?Ŕ8GT@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice???????!??j??@)???????1??j??@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch_?Q?{?!???????)_?Q?{?1???????:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zipn????!*:kǸ+P@)a??+ey?1oR?Wȍ??:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Rangea2U0*?c?!f??????)a2U0*?c?1f??????:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?J?4a?!@2??O??)?J?4a?1@2??O??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice??_?LU?!D?b?n??)??_?LU?1D?b?n??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI???V?"@Q+??"??V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?^lG^#%@רB??@2@!?o&???@	!       "$	????\Vd@???ձ?q@? 3??O\?!ѯ?_?~@*	!       2	!       :	?Ř?L@??$???%@!?
E??2@B	!       J	!       R	!       Z	!       b	!       JGPUb q???V?"@y+??"??V@