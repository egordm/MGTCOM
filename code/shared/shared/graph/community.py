from dataclasses import dataclass

from shared.graph import ComList, read_comlist, write_comlist, read_coms, coms_to_comlist, comlist_to_coms, write_coms, \
    Coms

NodeId = int


@dataclass
class CommunityAssignment:
    data: ComList

    @classmethod
    def load(cls, filepath: str) -> 'CommunityAssignment':
        return CommunityAssignment(
            read_comlist(filepath)
        )

    @classmethod
    def load_comms(cls, filepath: str) -> 'CommunityAssignment':
        return CommunityAssignment(
            coms_to_comlist(read_coms(filepath))
        )

    def write(self, filepath: str) -> None:
        write_comlist(self.data, filepath)

    def write_comms(self, filepath: str) -> None:
        write_coms(comlist_to_coms(self.data), filepath)

    def to_comms(self) -> Coms:
        return comlist_to_coms(self.data)
