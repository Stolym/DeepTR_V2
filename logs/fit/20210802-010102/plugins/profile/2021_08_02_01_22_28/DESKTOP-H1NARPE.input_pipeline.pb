$	??(??e@?\t?0s@F?̱??n?!?7??lv?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?7??lv?@o)狽S;@1?+?`?\~@Is-Z??U+@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails F?̱??n?!>???@`?1K?8???\?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!?v??/??1???]/Ma?I????G??r11*	ffffffw@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapP?s???!A?A4K@)??<,Ԛ??1Z??Y??G@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::MapR???Q??!?????_9@)[B>?٬??1MɔL??+@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatI.?!????!??????&@)?0?*???1?o??oy%@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::PrefetchHP?s??!t??s?w@)HP?s??1t??s?w@:Preprocessing2T
Iterator::Root::ParallelMapV2?z6?>??!	???@@)?z6?>??1	???@@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat=?U?????!?s??s?@)?I+???1??@:Preprocessing2E
Iterator::RootbX9?Ȧ?!Wx?Wx?'@)46<?R??1?dJ?dJ@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?J?4??!??N??^O@)?ZӼ?}?1x?Wx?W??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSliceǺ???f?!?~??~???)Ǻ???f?1?~??~???:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range??_vOf?!;?;???)??_vOf?1;?;???:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?J?4a?!.??-????)?J?4a?1.??-????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSliceǺ???V?!?~??~???)Ǻ???V?1?~??~???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?e>:?'@Q?\l?W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	K/Da*8"@??'??/@!o)狽S;@	!       "$	??( >d@??Rư?q@K?8???\?!?+?`?\~@*	!       2	!       :	u?0?F@?dr~??@!s-Z??U+@B	!       J	!       R	!       Z	!       b	!       JGPUb q?e>:?'@y?\l?W@