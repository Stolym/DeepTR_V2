$	8w?)?~f@????zs@?z?Fw??!??|Ͳހ@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'??|Ͳހ@7?X?Ou<@10.3?~@IbMeQ?-6@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ?z?Fw??a?xwd???1?t><K?a?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!?B;?Y???1??̔??b?Ix?=\r??r11*	?????}@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap?Fx$??!?+?+K@)??QI????1k`?k`?I@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Mapı.n???!??֊??@@)?q??۸?1I?H?4@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat???_vO??!????)@)-!?lV??1qe?pe?(@:Preprocessing2T
Iterator::Root::ParallelMapV2{?G?z??!??<??<@){?G?z??1??<??<@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat46<???!??????@)?q??????1????
@:Preprocessing2E
Iterator::Root2??%䃞?!y"?y"?@)n????1ѕ?Е? @:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::PrefetchU???N@??!P4P4 @)U???N@??1P4P4 @:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?n?????!&P?%P?M@)???Q?~?1[C?YC???:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSliceF%u?k?!????????)F%u?k?1????????:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??_vOf?!?ޝ?ޝ??)??_vOf?1?ޝ?ޝ??:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Rangeŏ1w-!_?!?3?3??)ŏ1w-!_?1?3?3??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlicea2U0*?S?!T?T???)a2U0*?S?1T?T???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?#????"@Q???K!?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	7?De??"@P~4?[m0@!7?X?Ou<@	!       "$	y??8.bd@-????q@?t><K?a?!0.3?~@*	!       2	!       :	}[?T?@????P?)@!bMeQ?-6@B	!       J	!       R	!       Z	!       b	!       JGPUb q?#????"@y???K!?V@