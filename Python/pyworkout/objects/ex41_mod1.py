from dataclasses import dataclass
from dataclasses import field as dcls_field
from typing import ClassVar


@dataclass
class Envelope:
    POSTAGE_MULTIPLIER: ClassVar[float] = dcls_field(default=10.0)

    weight: float
    postage: float = dcls_field(default=0.0)
    was_sent: bool = dcls_field(default=False)

    def send(
        self
    ) -> None:
        if self.postage >= self.postage_needed():
            self.was_sent = True

    def add_postage(
        self,
        postage: float
    ) -> None:
        self.postage = postage

    def postage_needed(
        self
    ) -> float:
        return self.weight * 10.0


@dataclass
class BigEnvelope(Envelope):
    POSTAGE_MULTIPLIER: ClassVar[float] = dcls_field(default=15.0)


def main():
    envelope = Envelope(50.0)
    print(
        envelope,
        envelope.POSTAGE_MULTIPLIER
    )

    bigenvelope = BigEnvelope(100.0)
    print(
        bigenvelope,
        bigenvelope.POSTAGE_MULTIPLIER
    )


if __name__ == '__main__':
    main()
