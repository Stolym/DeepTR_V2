$	?{x???e@??]???r@?I?%r???!?ͮ?]?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?ͮ?]?@?% ???<@1?V_]??}@Ik`???1@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ?I?%r??????o
+??1???מYb?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!???m????b????1??Z
H?_?r11*	?????au@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap????Mb??! 󛌽?K@)?.n????13d??:?H@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map-??臨?!?????9@)x$(~???1ʞ.?9,-@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat?N@aã?!bU6?&@) o?ŏ??1?@Ax$@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatV}??b??!??;`B@)???????1;4??f @:Preprocessing2E
Iterator::Root???~?:??!zA? X?"@)n????1?ܷ???@:Preprocessing2T
Iterator::Root::ParallelMapV2??@??ǈ?!?L?ںK@)??@??ǈ?1?L?ںK@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice??0?*??!?=?"?@)??0?*??1?=?"?@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch? ?	??!?vt?@)? ?	??1?vt?@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipH?}8g??!K? M7P@)9??v??z?1Ux?f??:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range"??u??q?!P?3?o??)"??u??q?1P?3?o??:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor_?Q?[?!??ڃ????)_?Q?[?1??ڃ????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlicea2U0*?S?!??!ls??)a2U0*?S?1??!ls??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI??]??!@Q??O???V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	1A??
#@?Ixcu0@???o
+??!?% ???<@	!       "$	l? o?c@??m:q@??Z
H?_?!?V_]??}@*	!       2	!       :	?????@~Em?I?$@!k`???1@B	!       J	!       R	!       Z	!       b	!       JGPUb q??]??!@y??O???V@