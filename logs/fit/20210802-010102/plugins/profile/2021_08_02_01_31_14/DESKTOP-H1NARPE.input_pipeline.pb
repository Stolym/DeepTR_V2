$	??
???e@?pm??r@???%V?!
??$?<?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'
??$?<?@y ?HK9@1Z???
z}@I(+???6@r0"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails???%V?1???%V?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!Qf?L2r??1??Z
H?_?I????}r??r11*	hffff^x@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapL?
F%u??!sEl??J@)?7??d???1??9?"?H@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map???9#J??!??e? W;@)????߮?1??|?V?.@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat??3????!??N???'@)?f??j+??1?J׃?5&@:Preprocessing2T
Iterator::Root::ParallelMapV2??&???!o??Nɲ@)??&???1o??Nɲ@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatV}??b??!`XZ???@)??A?f??1??G?p@:Preprocessing2E
Iterator::Root?J?4??!??h&?&@){?G?z??1A,:???@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip????߾??!ڲT??N@)"??u????1??'?B?@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::PrefetchF%u?{?!??Yx??)F%u?{?1??Yx??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice"??u??q?!??'?B???)"??u??q?1??'?B???:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????Mbp?!?V?dj??)????Mbp?1?V?dj??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlicea??+ei?!o???Bq??)a??+ei?1o???Bq??:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range?~j?t?h?!?y?'???)?~j?t?h?1?y?'???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?'?g?z"@Q????V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?̅?? @&/??4-@!y ?HK9@	!       "$	s?8约c@?._ñq@???%V?!Z???
z}@*	!       2	!       :	s??+L@}?S?/4*@!(+???6@B	!       J	!       R	!       Z	!       b	!       JGPUb q?'?g?z"@y????V@