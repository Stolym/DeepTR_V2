$	?d??&V@X??W%f@Ǻ???v?!m??~?%v@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'm??~?%v@? ???9@1,??WPt@IG?	1??@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ??.?????k$	?1iUMu_?r3"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsǺ???v?1Ǻ???v?r5"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!?fh<?????'ׄ?1?'eRC{?r11*	??????y@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap???V?/??!s???I@)??y???1R?8'?H@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?V-??!??9\`A@)vOjM??1TS?t2@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat??镲??!?????L0@)q???h??1???`/@:Preprocessing2T
Iterator::Root::ParallelMapV2?0?*???!"ϱOL?@)?0?*???1"ϱOL?@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat??y?):??!???l@)2U0*???1?1?A6?@:Preprocessing2E
Iterator::Root???Q???!6???Z^@)??ׁsF??1)tSRb@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::PrefetchF%u?{?!?Eo'???)F%u?{?1?Eo'???:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip????H??!U?@g??L@)-C??6z?1?Rʩ??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice?????g?!-{+?1???)?????g?1-{+?1???:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range{?G?zd?!ϰ??<???){?G?zd?1ϰ??<???:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?J?4a?!p??lGr??)?J?4a?1p??lGr??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice?~j?t?X?!?m|-?~??)?~j?t?X?1?m|-?~??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 7.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?T_??? @Qp?-
?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	{?(@?K'?)@!? ???9@	!       "$	b.??PT@?ظxDPd@iUMu_?!,??WPt@*	!       2	!       :	G?	1????G?	1?? @!G?	1??@B	!       J	!       R	!       Z	!       b	!       JGPUb q?T_??? @yp?-
?V@