$	#???Hd@??Y/?q@???͊?!??X3?Z~@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'??X3?Z~@$????z<@1K %v-?{@I29?3L?0@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ???͊?mU?Y??1fL?g?a?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!5?|?????1???]/Mq?I??????r11*	gffff?@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapm???????!A?4?T?I@)?MbX9??1?wd?9G@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::MapW[??????!??,?=@){?/L?
??1?0 Ό/@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat333333??!?&?M?{+@)?&S???1M?+?*@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?46<??!?v??d?$@)ı.n???1?ZF?A$@:Preprocessing2T
Iterator::Root::ParallelMapV2??JY?8??!?o?Bx?@)??JY?8??1?o?Bx?@:Preprocessing2E
Iterator::Root0L?
F%??!??+?zD@)n????1??#}?@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice??H?}??!???CL@)??H?}??1???CL@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch?&S???!M?+???)?&S???1M?+???:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zipё\?C???!?tVZuO@)?q?????1a?螒???:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora2U0*?c?!?{??e$??)a2U0*?c?1?{??e$??:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range/n??b?!@1;????)/n??b?1@1;????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice?~j?t?X?!7-???)?~j?t?X?17-???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?d????"@Qe??"?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	J+??"@???^?p0@!$????z<@	!       "$	???/\b@e?&???o@fL?g?a?!K %v-?{@*	!       2	!       :	?84R܎@&Q?h?"@!29?3L?0@B	!       J	!       R	!       Z	!       b	!       JGPUb q?d????"@ye??"?V@