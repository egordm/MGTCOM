from dataclasses import dataclass
from typing import Optional

from neo4j import GraphDatabase, BoltDriver
from simple_parsing import Serializable

from shared.constants import GLOBAL_CONFIG_PATH, BENCHMARKS_LOGS
from shared.logger import get_logger

LOG = get_logger('config')


@dataclass
class Neo4JConfig(Serializable):
    host: str = 'localhost'
    port: int = '7687'
    username: str = 'neo4j'
    password: str = 'test'
    database: str = 'neo4j'
    java_home: Optional[str] = None
    neo4j_home: Optional[str] = None

    def open(self, check_connection=True) -> BoltDriver:
        driver = GraphDatabase.driver(f'bolt://{self.host}:{self.port}', auth=(self.username, self.password))
        if check_connection:
            driver.verify_connectivity()
        return driver


wandb_initialized = False


@dataclass
class WandbConfig(Serializable):
    api_key: str = ''
    mode: str = 'online'

    def open(self, **kwargs) -> None:
        global wandb_initialized
        if not wandb_initialized:
            LOG.info('Initializing wandb')
            import wandb

            if self.mode == 'online':
                LOG.info('wadb Running in online mode')
                wandb.login(key=self.api_key)

            BENCHMARKS_LOGS.mkdir(exist_ok=True, parents=True)
            wandb.init(
                mode=self.mode,
                dir=BENCHMARKS_LOGS,
                project='Thesis',
                **kwargs
            )
            wandb_initialized = True


@dataclass
class ConnectionConfig(Serializable):
    neo4j: Neo4JConfig = Neo4JConfig()
    wandb: WandbConfig = WandbConfig()

    @staticmethod
    def load_config(path=None) -> 'ConnectionConfig':
        path = path or GLOBAL_CONFIG_PATH.joinpath('connections.yml')
        return ConnectionConfig.load(path)
