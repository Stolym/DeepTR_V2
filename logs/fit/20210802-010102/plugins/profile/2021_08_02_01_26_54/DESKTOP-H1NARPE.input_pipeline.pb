$	Y?LW?e@2??K?r@??H?}??!?.??s?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?.??s?@??$??<@1OYM?S?}@IR(__?3@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ?z?????ꫫ???1?@fg?;e?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!??H?}??1:?`???d?ID???XP??r11*	33333?y@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap?@??ǘ??!??1͠?I@)?Q?????1\
?I?bH@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?w??#???!<8?H?;@)?Q?????1?????0@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat^K?=???!F?4??S$@)8??d?`??1???Y?+#@:Preprocessing2E
Iterator::RootF%u???!U????n)@)vOjM??1?????("@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat9??v????!{J?T@)Zd;?O???1??yor(@:Preprocessing2T
Iterator::Root::ParallelMapV2?W[?????!??h+@)?W[?????1??h+@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?c]?F??!M|?wK!N@)"??u????1??? ? @:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetchvq?-??!n???rq??)vq?-??1n???rq??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice?q????o?!??E5???)?q????o?1??E5???:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?~j?t?h?!7??<??)?~j?t?h?17??<??:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Rangea2U0*?c?!????B??)a2U0*?c?1????B??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice??_?LU?!b?x?	??)??_?LU?1b?x?	??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?*??^"@Q?:??(?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?
?-11#@?u?[F?0@!??$??<@	!       "$	!??E?c@??m?@q@:?`???d?!OYM?S?}@*	!       2	!       :	?1???@~??	C?&@!R(__?3@B	!       J	!       R	!       Z	!       b	!       JGPUb q?*??^"@y?:??(?V@