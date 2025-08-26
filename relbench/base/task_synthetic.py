from typing import Optional

import duckdb
import pandas as pd

from tqdm import tqdm

from relbench.metrics import (
    accuracy,
    average_precision,
    f1,
    mae,
    r2,
    rmse,
    roc_auc,
)
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph

from .database import Database
from .dataset import Dataset
from .table import Table
from .task_base import TaskType
from .task_entity import EntityTask

from typing import Any, Dict, List

import torch
from torch import Tensor
from torch.nn import Embedding, ModuleDict
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType
from torch_geometric.loader import NeighborLoader

from relbench.modeling.utils import get_stype_proposal
from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder
from typing import List, Optional

import torch
from torch_frame.config.text_embedder import TextEmbedderConfig
from sentence_transformers import SentenceTransformer
from torch import Tensor


class GloveTextEmbedding:
    def __init__(self, device: Optional[torch.device] = None):
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d",
            device=device,
        )

    def __call__(self, sentences: List[str]) -> Tensor:
        return self.model.encode(sentences, convert_to_tensor=True)

class Model(torch.nn.Module):
    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        num_layers: int,
        channels: int,
        out_channels: int,
        aggr: str,
        norm: str,
        # List of node types to add shallow embeddings to input
        shallow_list: List[NodeType] = [],
        # ID awareness
        id_awareness: bool = False,
    ):
        super().__init__()

        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
            torch_frame_model_kwargs = {
                "channels": 128,
                "num_layers": 4,
                "dropout_prob": 0.0,
            },
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[
                node_type for node_type in data.node_types if "time" in data[node_type]
            ],
            channels=channels,
        )
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers,
        )
        self.head = MLP(
            channels,
            out_channels=out_channels,
            norm=norm,
            num_layers=1,
        )
        self.embedding_dict = ModuleDict(
            {
                node: Embedding(data.num_nodes_dict[node], channels)
                for node in shallow_list
            }
        )

        self.id_awareness_emb = None
        if id_awareness:
            self.id_awareness_emb = torch.nn.Embedding(1, channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()
        for embedding in self.embedding_dict.values():
            torch.nn.init.normal_(embedding.weight, std=0.1)
        if self.id_awareness_emb is not None:
            self.id_awareness_emb.reset_parameters()

    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType,
    ) -> Tensor:
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )

        return self.head(x_dict[entity_table][: seed_time.size(0)])

    def forward_dst_readout(
        self,
        batch: HeteroData,
        entity_table: NodeType,
        dst_table: NodeType,
    ) -> Tensor:
        if self.id_awareness_emb is None:
            raise RuntimeError(
                "id_awareness must be set True to use forward_dst_readout"
            )
        seed_time = batch[entity_table].seed_time
        with torch.no_grad():
            x_dict = self.encoder(batch.tf_dict)
        # Add ID-awareness to the root node
        x_dict[entity_table][: seed_time.size(0)] += self.id_awareness_emb.weight

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
        )

        return self.head(x_dict[dst_table])



class SyntheticTask(EntityTask):
    timedelta = pd.Timedelta(seconds=1)
    entity_col: str

    def __init__(
        self,
        dataset: Dataset,
        task_type: TaskType,
        entity_table: str,
        num_layers: int = 2,
        channels: int = 64,
        num_neighbors: int = 128,
        aggr: str = "sum",
        norm: str = "batch_norm",
        timedelta: pd.Timedelta = pd.Timedelta(seconds=1),
        temporal_strategy: str = "uniform",
        cache_dir: Optional[str] = None,
        device: str = "cpu",
    ):
        super().__init__(dataset, cache_dir=cache_dir)

        self.task_type = task_type
        self.entity_table = entity_table
        self.num_layers = num_layers
        self.channels = channels
        self.num_neighbors = num_neighbors
        self.aggr = aggr
        self.norm = norm
        self.timedelta = timedelta
        self.temporal_strategy = temporal_strategy
        self.device = device    

        db = self.dataset.get_db(upto_test_timestamp=False) # TODO: verify this is ok
        entity_col = db.table_dict[entity_table].pkey_col
        self.entity_col = entity_col if entity_col is not None else "primary_key"
        self.time_col = db.table_dict[entity_table].time_col

        self.target_col = "target"  # Placeholder for target column
        
        if self.task_type == TaskType.REGRESSION:
            self.metrics = [r2, mae, rmse]
        elif self.task_type == TaskType.BINARY_CLASSIFICATION:
            self.metrics = [average_precision, accuracy, f1, roc_auc]
        else:
            raise NotImplementedError(f"Task type {self.task_type} not implemented")

        # Prepare the data        
        self.col_to_stype_dict = get_stype_proposal(db)

        self.data, col_stats_dict = make_pkey_fkey_graph(
            db,
            col_to_stype_dict=self.col_to_stype_dict,
            text_embedder_cfg=TextEmbedderConfig(
                text_embedder=GloveTextEmbedding(device=self.device), batch_size=256
            ),
        )
        
        # Create the model (random GNN)
        # TODO: simply this by removing torch_frame encoders
        self.model = Model(
            data=self.data,
            col_stats_dict=col_stats_dict,
            num_layers=self.num_layers,
            channels=self.channels,
            out_channels=1,
            aggr=self.aggr,
            norm=self.norm,
        ).to(self.device)

    def _get_table(self, split: str) -> Table:
        r"""Helper function to get a table for a split.

        This function overrides the `_get_table` method in `EntityTask`.
        Because we predict all values in the target column, we only look at the min and max timestamp
        for each split and take all rows in the table between them.
        """

        db = self.dataset.get_db(upto_test_timestamp=split != "test")
        # NOTE: (1) As opposed to with standard relbench tasks we do not move back in time here.
        # We use the original dates (with the timedelta as the frequency) and compute a transformation
        # of the graph at that point in time. Only after the transform, we move back in time.
        if split == "train":
            start = self.dataset.val_timestamp
            end = db.min_timestamp
            freq = -self.timedelta

        elif split == "val":
            if self.dataset.val_timestamp + self.timedelta > db.max_timestamp:
                raise RuntimeError(
                    "val timestamp + timedelta is larger than max timestamp! "
                    "This would cause val labels to be generated with "
                    "insufficient aggregation time."
                )

            start = self.dataset.test_timestamp
            end = self.dataset.val_timestamp
            freq = -self.timedelta

        elif split == "test":
            if self.dataset.test_timestamp + self.timedelta > db.max_timestamp:
                raise RuntimeError(
                    "test timestamp + timedelta is larger than max timestamp! "
                    "This would cause test labels to be generated with "
                    "insufficient aggregation time."
                )

            start = db.max_timestamp
            end = self.dataset.test_timestamp
            freq = -self.timedelta

        timestamps = pd.date_range(start=start, end=end, freq=freq)

        if split == "train" and len(timestamps) < 3:
            raise RuntimeError(
                f"The number of training time frames is too few. "
                f"({len(timestamps)} given)"
            )

        
        self.split = split
        table = self.make_table(db, timestamps)
        # FIXME: this filter could be a problem in autocomplete tasks!!
        # table = self.filter_dangling_entities(table)

        return table

    def make_train_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        entity_table = db.table_dict[self.entity_table].df  # noqa: F841

        # Calculate minimum and maximum timestamps from timestamp_df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        min_timestamp = timestamp_df["timestamp"].min()
        max_timestamp = timestamp_df["timestamp"].max()

        df = duckdb.sql(
            f"""
            SELECT
                entity_table.{self.time_col},
                entity_table.{self.entity_col}
            FROM
                entity_table
            WHERE
                entity_table.{self.time_col} > '{min_timestamp}' AND
                entity_table.{self.time_col} <= '{max_timestamp}'
            """
        ).df()
        return Table(
            df=df,
            fkey_col_to_pkey_table={
                self.entity_col: self.entity_table,
            },
            pkey_col=None,
            time_col=self.time_col,
        )

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:

        table = self.make_train_table(db, timestamps)        
        table_input = get_node_train_table_input(table=table, task=self)
        
        if self.split == "test":
            pass

        if self.num_neighbors == -1:
            num_neighbors = [-1 for _ in range(self.num_layers)]
        else:
            num_neighbors = [int(self.num_neighbors / 2**i) for i in range(self.num_layers)]
        dataloader = NeighborLoader(
            self.data,
            num_neighbors=num_neighbors,
            time_attr="time",
            input_nodes=table_input.nodes,
            input_time=table_input.time,
            transform=table_input.transform,
            batch_size=128,
            temporal_strategy=self.temporal_strategy,
            shuffle=False,
            num_workers=0,
            disjoint=False,
            # persistent_workers=args.num_workers > 0,
        )

        with torch.no_grad():
            self.model.eval()
            pred_list = []
            ids = []
            for batch in tqdm(dataloader):
                batch = batch.to(self.device)
                pred = self.model(
                    batch,
                    self.entity_table,
                )

                pred = pred.view(-1) if pred.size(1) == 1 else pred
                pred_list.append(pred.detach().cpu())
                ids.append(batch[self.entity_table].n_id.detach().cpu())
            preds = torch.cat(pred_list, dim=0).numpy()
            ids = torch.cat(ids, dim=0).numpy()
            if self.task_type == TaskType.REGRESSION:
                pass # TODO: can implement additional transforms if needed
            elif self.task_type == TaskType.BINARY_CLASSIFICATION:
                preds = (preds > preds.mean()).astype(int)
            else:
                raise NotImplementedError(
                    f"Task type {self.task_type} is not implemented"
                )
        
        df = table.df.copy()
        df[self.target_col] = preds
        # NOTE: (2) Now we move back in time, as opposed to with standard relbench tasks.
        df[self.time_col] = df[self.time_col] - self.timedelta
        return Table(
            df=df,
            fkey_col_to_pkey_table={
                self.entity_col: self.entity_table,
            },
            pkey_col=None,
            time_col=self.time_col,
        )
