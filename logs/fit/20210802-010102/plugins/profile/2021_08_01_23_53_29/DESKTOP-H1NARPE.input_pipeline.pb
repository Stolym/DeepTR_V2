$	G?rT?f@s???s@????y??!??????@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'??????@??[7@1~V?)?~@I?{L? 8@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ??	L?u????v???1rQ-"??[?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!????y??????y7??1C?8
a?r11*	?????,}@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap8gDio??!c??.??K@)O??e?c??1k?k͗H@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Mapޓ??ZӼ?!???c&8@)	?c?Z??1y[1qD?.@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatt$???~??!?Ek?'@)??A?f??1??|I??!@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat??D????!l??U?!@)@?߾???1$S?]?h @:Preprocessing2E
Iterator::RootL7?A`???!???G@)Q?|a2??1]???߼@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice???QI??!=?W݁@)???QI??1=?W݁@:Preprocessing2T
Iterator::Root::ParallelMapV2?(??0??!7	ʩf@)?(??0??17	ʩf@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensortF??_??!?!??d@)tF??_??1?!??d@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetchg??j+???!??u)@)g??j+???1??u)@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?i?q????!??5?3Q@)'???????1?c&nV@:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range??_?Le?!??'?????)??_?Le?1??'?????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlicea2U0*?S?!?????s??)a2U0*?S?1?????s??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?f3?)?!@Q#?????V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	???g׺@??????*@????y7??!??[7@	!       "$	nYk?d@J???l`q@rQ-"??[?!~V?)?~@*	!       2	!       :	9W??? @?%?+?+@!?{L? 8@B	!       J	!       R	!       Z	!       b	!       JGPUb q?f3?)?!@y#?????V@