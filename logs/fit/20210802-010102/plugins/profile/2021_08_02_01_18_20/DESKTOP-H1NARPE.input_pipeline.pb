$	? ??Og@F???/t@?????щ?!??U??{?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'??U??{?@E.8??G=@1????@I?qQ-"?7@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ?????щ??OU??X??1rQ-"??[?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!؃I??	??1?t><K?a?I9{????r11*	?????iu@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap?Fx$??!?U?Y?K@)0?'???1& ?}??I@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map$(~????!}t'x??@)ı.n???1yr???/@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatS?!?uq??!??E=FJ/@) ?o_Ω?1???l-@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat??ܵ?|??!?c?v$?@)???S㥋?1N?8s?@:Preprocessing2T
Iterator::Root::ParallelMapV2?(??0??!?*?붸@)?(??0??1?*?붸@:Preprocessing2E
Iterator::RootHP?s??!5i@???@)?g??s???1ԧ?V|?@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetchlxz?,C|?!?0??? @)lxz?,C|?1?0??? @:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipW?/?'??!???0?N@)S?!?uq{?1??E=FJ??:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range-C??6j?!????????)-C??6j?1????????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice??_vOf?!r???
8??)??_vOf?1r???
8??:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??_?Le?!5\???H??)??_?Le?15\???H??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice-C??6Z?!????????)-C??6Z?1????????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI??Z???"@Q!?????V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????#@?.???0@!E.8??G=@	!       "$	a?l?e@?1?h?Fr@rQ-"??[?!????@*	!       2	!       :	AF@?#?@?????N+@!?qQ-"?7@B	!       J	!       R	!       Z	!       b	!       JGPUb q??Z???"@y!?????V@