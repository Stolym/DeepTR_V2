$	???m<uf@m?-	Xqs@?IbI????!?Q??ր@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?Q??ր@A?
?t=@1???d?~@I/???4@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ?IbI?????P?l??1rQ-"??[?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!J?_????1?@fg?;e?I@l??TO??r11*	??????v@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap?8??m4??!?\?(?%J@)???_vO??1??ZX?H@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map5?8EGr??!??S+};@)?rh??|??1?x?1@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat?k	??g??!?e?&_?$@)?V-??1?(?u??#@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatŏ1w-!??!?M?l? @)??6???1a??V?@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zipd;?O????!ja???P@)+??????1?!???@:Preprocessing2T
Iterator::Root::ParallelMapV2tF??_??!,=???S
@)tF??_??1,=???S
@:Preprocessing2E
Iterator::Root???????!??ZX??@)M??St$??1      	@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch??ׁsF??!Fq?c?@)??ׁsF??1Fq?c?@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice??H?}m?!	???????)??H?}m?1	???????:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorF%u?k?!?A?/4??)F%u?k?1?A?/4??:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Rangea2U0*?c?!??S+=??)a2U0*?c?1??S+=??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice??_?LU?!?$??C??)??_?LU?1?$??C??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?s?
??"@Q?Q??C?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	v@?*??#@;qᦸ?0@!A?
?t=@	!       "$	v)???^d@.s??=?q@rQ-"??[?!???d?~@*	!       2	!       :	1??? o@???}ާ'@!/???4@B	!       J	!       R	!       Z	!       b	!       JGPUb q?s?
??"@y?Q??C?V@