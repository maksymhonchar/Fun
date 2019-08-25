# src: https://www.youtube.com/watch?v=QGgAFG2a7Ic

from mrjob.job import MRJob
from mrjob.step import MRStep


class RatingsBreakdown(MRJob):
    
    def steps(self):
        return [
            MRStep(
                mapper=self._mapper_get_ratings,
                reducer=self._reducer_count_ratings
            )
        ]

    def _mapper_get_ratings(self, _, line):
        (uid, mid, rating, timestamp) = line.split('\t')
        yield (rating, 1)

    def _reducer_count_ratings(self, hkey, hvalues):
        yield (hkey, sum(hvalues))
    

RatingsBreakdown.run()
