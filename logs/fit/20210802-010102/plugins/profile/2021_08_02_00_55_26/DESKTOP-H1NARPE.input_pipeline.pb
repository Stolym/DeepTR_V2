$	o?G??f@v?(v??s@b?o???!??Z?<.?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'??Z?<.?@?~k'J?:@1s????f@I???a?4@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails b?o???6w??\???1?t><K?a?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!??m?2??1a2U0*?c?I??q?@H??r11*	fffff.x@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapz6?>W[??!??X?֙I@)???B?i??1?k??F@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Mapk?w??#??!X????a9@)1?Zd??1?jڗݧ+@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat?,C????!?ς?S'@)??_?L??1?
A}?%@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?:pΈ??!v?9}?"@)?St$????1י??|)!@:Preprocessing2T
Iterator::Root::ParallelMapV2tF??_??!	?ej?@)tF??_??1	?ej?@:Preprocessing2E
Iterator::RootM??St$??!U??q~]'@)'???????1??{y?@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice???????!/+nK?@)???????1/+nK?@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip^?I+??!>???wO@)HP?sׂ?1cW??@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch?J?4??!??e?k^@)?J?4??1??e?k^@:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Rangea??+ei?!?Qdã??)a??+ei?1?Qdã??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice?~j?t?h?!??#i???)?~j?t?h?1??#i???:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?~j?t?h?!??#i???)?~j?t?h?1??#i???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?*?y;!@Q?ڇܐ?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??W?g?!@?????.@!?~k'J?:@	!       "$	)q0?K?d@?YX#9!r@?t><K?a?!s????f@*	!       2	!       :	?6خ?r@^????'@!???a?4@B	!       J	!       R	!       Z	!       b	!       JGPUb q?*?y;!@y?ڇܐ?V@