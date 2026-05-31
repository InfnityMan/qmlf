import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import torch


class QVizPro:
    @staticmethod
    def _to_numpy(data):
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return np.asarray(data)

    @staticmethod
    def _show_fig(fig, show=True, save_path=None):
        if save_path is not None:
            fig.write_html(save_path)
        if show:
            fig.show()
        return fig

    @staticmethod
    def _safe_perplexity(n_samples, perplexity):
        if n_samples <= 3:
            return None
        return min(perplexity, n_samples - 1)

    @staticmethod
    def _pad_to_3d(embedding):
        if embedding.shape[1] == 3:
            return embedding

        padded = np.zeros((embedding.shape[0], 3))
        padded[:, :embedding.shape[1]] = embedding
        return padded

    @staticmethod
    def _kernel_to_distance(kernel_matrix):
        kernel_matrix = (kernel_matrix + kernel_matrix.T) / 2
        diagonal = np.diag(kernel_matrix)
        distances = diagonal[:, None] + diagonal[None, :] - 2 * kernel_matrix
        distances = np.maximum(distances, 0)
        distances = np.sqrt(distances)
        np.fill_diagonal(distances, 0)
        return distances

    @staticmethod
    def _state_vectors_to_bloch(state_vectors):
        state_vectors = QVizPro._to_numpy(state_vectors)

        if state_vectors.ndim != 2:
            raise ValueError("state_vectors must be a 2D array")

        if state_vectors.shape[1] == 3:
            return state_vectors.real

        if state_vectors.shape[1] != 2:
            raise ValueError(
                "state_vectors must be 3D Bloch coordinates or 2D single-qubit state vectors"
            )

        alpha = state_vectors[:, 0]
        beta = state_vectors[:, 1]

        norm = np.sqrt(np.abs(alpha) ** 2 + np.abs(beta) ** 2)
        norm[norm == 0] = 1

        alpha = alpha / norm
        beta = beta / norm

        x = 2 * np.real(np.conj(alpha) * beta)
        y = 2 * np.imag(np.conj(alpha) * beta)
        z = np.abs(alpha) ** 2 - np.abs(beta) ** 2

        return np.column_stack([x, y, z])

    @staticmethod
    def plot_hilbert_space(
        kernel_matrix,
        labels=None,
        method="tsne",
        perplexity=30,
        learning_rate=200,
        title="Quantum Hilbert Space Visualization",
        show=True,
        save_path=None
    ):
        kernel_matrix = QVizPro._to_numpy(kernel_matrix)

        if kernel_matrix.ndim != 2:
            raise ValueError("kernel_matrix must be a 2D matrix")

        n_samples = kernel_matrix.shape[0]

        if n_samples < 2:
            raise ValueError("At least 2 samples are required")

        if labels is not None:
            labels = QVizPro._to_numpy(labels).flatten()
            if len(labels) != n_samples:
                raise ValueError("labels must match the number of samples")

        is_square_kernel = kernel_matrix.shape[0] == kernel_matrix.shape[1]

        if method == "tsne" and n_samples > 3:
            safe_perplexity = QVizPro._safe_perplexity(n_samples, perplexity)

            if is_square_kernel:
                distances = QVizPro._kernel_to_distance(kernel_matrix)

                tsne = TSNE(
                    n_components=3,
                    random_state=42,
                    perplexity=safe_perplexity,
                    learning_rate=learning_rate,
                    metric="precomputed",
                    init="random"
                )

                embedding = tsne.fit_transform(distances)

            else:
                tsne = TSNE(
                    n_components=3,
                    random_state=42,
                    perplexity=safe_perplexity,
                    learning_rate=learning_rate
                )

                embedding = tsne.fit_transform(kernel_matrix)

            projection_name = "t-SNE Projection"

        else:
            n_components = min(3, kernel_matrix.shape[0], kernel_matrix.shape[1])

            pca = PCA(n_components=n_components)
            embedding = pca.fit_transform(kernel_matrix)
            embedding = QVizPro._pad_to_3d(embedding)

            projection_name = "PCA Projection"

        fig = go.Figure(data=[
            go.Scatter3d(
                x=embedding[:, 0],
                y=embedding[:, 1],
                z=embedding[:, 2],
                mode="markers",
                marker=dict(
                    size=7,
                    color=labels if labels is not None else embedding[:, 0],
                    colorscale="Viridis",
                    opacity=0.85,
                    line=dict(width=0.8, color="DarkSlateGrey")
                ),
                name="Samples"
            )
        ])

        fig.update_layout(
            title=f"{title} ({projection_name})",
            scene=dict(
                xaxis_title="Dimension 1",
                yaxis_title="Dimension 2",
                zaxis_title="Dimension 3",
                aspectmode="cube"
            ),
            height=720
        )

        return QVizPro._show_fig(fig, show, save_path)

    @staticmethod
    def plot_kernel_matrix(
        kernel_matrix,
        labels=None,
        title="Quantum Kernel Matrix",
        show=True,
        save_path=None
    ):
        kernel_matrix = QVizPro._to_numpy(kernel_matrix)

        if kernel_matrix.ndim != 2:
            raise ValueError("kernel_matrix must be a 2D matrix")

        if kernel_matrix.shape[0] != kernel_matrix.shape[1]:
            raise ValueError("kernel_matrix must be square")

        if labels is not None:
            labels = QVizPro._to_numpy(labels).flatten()

            if len(labels) != kernel_matrix.shape[0]:
                raise ValueError("labels must match kernel size")

        axis_labels = labels if labels is not None else np.arange(kernel_matrix.shape[0])

        fig = go.Figure(data=go.Heatmap(
            z=kernel_matrix,
            x=axis_labels,
            y=axis_labels,
            colorscale="Viridis",
            colorbar=dict(title="Kernel Value")
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Sample",
            yaxis_title="Sample",
            height=650
        )

        return QVizPro._show_fig(fig, show, save_path)

    @staticmethod
    def plot_kernel_eigenvalues(
        kernel_matrix,
        title="Quantum Kernel Eigenvalue Spectrum",
        show=True,
        save_path=None
    ):
        kernel_matrix = QVizPro._to_numpy(kernel_matrix)

        if kernel_matrix.ndim != 2:
            raise ValueError("kernel_matrix must be a 2D matrix")

        if kernel_matrix.shape[0] != kernel_matrix.shape[1]:
            raise ValueError("kernel_matrix must be square")

        kernel_matrix = (kernel_matrix + kernel_matrix.T) / 2

        eigenvalues = np.linalg.eigvalsh(kernel_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]

        fig = go.Figure(data=[
            go.Bar(
                x=np.arange(1, len(eigenvalues) + 1),
                y=eigenvalues,
                name="Eigenvalues"
            )
        ])

        fig.update_layout(
            title=title,
            xaxis_title="Eigenvalue Rank",
            yaxis_title="Eigenvalue",
            height=520
        )

        return QVizPro._show_fig(fig, show, save_path)

    @staticmethod
    def gradient_heatmap(
        gradients,
        title="Gradient Magnitude Across Parameters",
        show_values=True,
        show=True,
        save_path=None
    ):
        gradients = QVizPro._to_numpy(gradients)

        if gradients.ndim == 1:
            gradients = gradients.reshape(1, -1)

        if gradients.ndim != 2:
            raise ValueError("gradients must be a 1D or 2D array")

        heatmap_args = dict(
            z=np.abs(gradients),
            colorscale="RdBu_r",
            colorbar=dict(title="|Gradient|")
        )

        if show_values and gradients.size <= 250:
            heatmap_args["text"] = np.round(gradients, 4)
            heatmap_args["texttemplate"] = "%{text}"

        fig = go.Figure(data=go.Heatmap(**heatmap_args))

        fig.update_layout(
            title=title,
            xaxis_title="Parameters",
            yaxis_title="Samples / Circuits",
            height=600
        )

        return QVizPro._show_fig(fig, show, save_path)

    @staticmethod
    def plot_gradient_variance(
        gradients,
        title="Gradient Variance Across Parameters",
        log_scale=True,
        show=True,
        save_path=None
    ):
        gradients = QVizPro._to_numpy(gradients)

        if gradients.ndim == 1:
            gradients = gradients.reshape(1, -1)

        if gradients.ndim != 2:
            raise ValueError("gradients must be a 1D or 2D array")

        variances = np.var(gradients, axis=0)
        means = np.mean(np.abs(gradients), axis=0)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=np.arange(len(variances)),
            y=variances,
            mode="lines+markers",
            name="Gradient Variance"
        ))

        fig.add_trace(go.Scatter(
            x=np.arange(len(means)),
            y=means,
            mode="lines+markers",
            name="Mean Absolute Gradient"
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Parameter Index",
            yaxis_title="Gradient Statistic",
            height=540
        )

        if log_scale:
            fig.update_yaxes(type="log")

        return QVizPro._show_fig(fig, show, save_path)

    @staticmethod
    def plot_gradient_distribution(
        gradients,
        title="Gradient Distribution",
        bins=40,
        show=True,
        save_path=None
    ):
        gradients = QVizPro._to_numpy(gradients).flatten()

        fig = go.Figure(data=[
            go.Histogram(
                x=gradients,
                nbinsx=bins,
                name="Gradients"
            )
        ])

        fig.add_vline(
            x=0,
            line_width=2,
            line_dash="dash"
        )

        fig.update_layout(
            title=title,
            xaxis_title="Gradient Value",
            yaxis_title="Count",
            height=520
        )

        return QVizPro._show_fig(fig, show, save_path)

    @staticmethod
    def plot_barren_plateau(
        gradient_variances,
        depths=None,
        title="Barren Plateau Diagnostic",
        log_scale=True,
        show=True,
        save_path=None
    ):
        gradient_variances = QVizPro._to_numpy(gradient_variances).flatten()

        if depths is None:
            depths = np.arange(1, len(gradient_variances) + 1)
        else:
            depths = QVizPro._to_numpy(depths).flatten()

        if len(depths) != len(gradient_variances):
            raise ValueError("depths and gradient_variances must have the same length")

        fig = go.Figure(data=[
            go.Scatter(
                x=depths,
                y=gradient_variances,
                mode="lines+markers",
                marker=dict(size=8),
                name="Gradient Variance"
            )
        ])

        fig.update_layout(
            title=title,
            xaxis_title="Circuit Depth / Ansatz Repetitions",
            yaxis_title="Gradient Variance",
            height=540
        )

        if log_scale:
            fig.update_yaxes(type="log")

        return QVizPro._show_fig(fig, show, save_path)

    @staticmethod
    def plot_bloch_sphere(
        state_vectors=None,
        title="Quantum States on Bloch Sphere",
        show_vectors=True,
        show=True,
        save_path=None
    ):
        fig = go.Figure()

        u = np.linspace(0, 2 * np.pi, 60)
        v = np.linspace(0, np.pi, 60)

        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))

        fig.add_trace(go.Surface(
            x=x,
            y=y,
            z=z,
            opacity=0.25,
            colorscale="Blues",
            showscale=False,
            name="Bloch Sphere"
        ))

        fig.add_trace(go.Scatter3d(
            x=[-1, 1],
            y=[0, 0],
            z=[0, 0],
            mode="lines",
            line=dict(width=4),
            name="X Axis"
        ))

        fig.add_trace(go.Scatter3d(
            x=[0, 0],
            y=[-1, 1],
            z=[0, 0],
            mode="lines",
            line=dict(width=4),
            name="Y Axis"
        ))

        fig.add_trace(go.Scatter3d(
            x=[0, 0],
            y=[0, 0],
            z=[-1, 1],
            mode="lines",
            line=dict(width=4),
            name="Z Axis"
        ))

        if state_vectors is not None:
            bloch_points = QVizPro._state_vectors_to_bloch(state_vectors)

            fig.add_trace(go.Scatter3d(
                x=bloch_points[:, 0],
                y=bloch_points[:, 1],
                z=bloch_points[:, 2],
                mode="markers",
                marker=dict(
                    size=9,
                    color=np.arange(len(bloch_points)),
                    colorscale="Plasma",
                    opacity=0.9
                ),
                name="Quantum States"
            ))

            if show_vectors:
                for point in bloch_points:
                    fig.add_trace(go.Scatter3d(
                        x=[0, point[0]],
                        y=[0, point[1]],
                        z=[0, point[2]],
                        mode="lines",
                        line=dict(width=3),
                        showlegend=False
                    ))

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                xaxis=dict(range=[-1.15, 1.15]),
                yaxis=dict(range=[-1.15, 1.15]),
                zaxis=dict(range=[-1.15, 1.15]),
                aspectmode="cube"
            ),
            height=720
        )

        return QVizPro._show_fig(fig, show, save_path)

    @staticmethod
    def plot_parameter_landscape(
        loss_values,
        x_values=None,
        y_values=None,
        title="Loss Landscape in Parameter Space",
        show_contours=True,
        show=True,
        save_path=None
    ):
        loss_values = QVizPro._to_numpy(loss_values)

        if loss_values.ndim != 2:
            raise ValueError("loss_values must be a 2D array")

        if x_values is None:
            x_values = np.arange(loss_values.shape[1])
        else:
            x_values = QVizPro._to_numpy(x_values).flatten()

        if y_values is None:
            y_values = np.arange(loss_values.shape[0])
        else:
            y_values = QVizPro._to_numpy(y_values).flatten()

        if len(x_values) != loss_values.shape[1]:
            raise ValueError("x_values must match loss_values columns")

        if len(y_values) != loss_values.shape[0]:
            raise ValueError("y_values must match loss_values rows")

        fig = go.Figure(data=[
            go.Surface(
                x=x_values,
                y=y_values,
                z=loss_values,
                colorscale="Viridis",
                contours=dict(
                    z=dict(
                        show=show_contours,
                        usecolormap=True,
                        highlightcolor="limegreen",
                        project_z=True
                    )
                )
            )
        ])

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="Parameter 1",
                yaxis_title="Parameter 2",
                zaxis_title="Loss"
            ),
            height=720
        )

        return QVizPro._show_fig(fig, show, save_path)

    @staticmethod
    def plot_training_history(
        losses,
        accuracies=None,
        val_losses=None,
        val_accuracies=None,
        title="QML Training History",
        show=True,
        save_path=None
    ):
        losses = QVizPro._to_numpy(losses).flatten()
        epochs = np.arange(1, len(losses) + 1)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=epochs,
            y=losses,
            mode="lines+markers",
            name="Training Loss"
        ))

        if val_losses is not None:
            val_losses = QVizPro._to_numpy(val_losses).flatten()

            if len(val_losses) != len(losses):
                raise ValueError("val_losses must match losses")

            fig.add_trace(go.Scatter(
                x=epochs,
                y=val_losses,
                mode="lines+markers",
                name="Validation Loss"
            ))

        if accuracies is not None:
            accuracies = QVizPro._to_numpy(accuracies).flatten()

            if len(accuracies) != len(losses):
                raise ValueError("accuracies must match losses")

            fig.add_trace(go.Scatter(
                x=epochs,
                y=accuracies,
                mode="lines+markers",
                name="Training Accuracy",
                yaxis="y2"
            ))

        if val_accuracies is not None:
            val_accuracies = QVizPro._to_numpy(val_accuracies).flatten()

            if len(val_accuracies) != len(losses):
                raise ValueError("val_accuracies must match losses")

            fig.add_trace(go.Scatter(
                x=epochs,
                y=val_accuracies,
                mode="lines+markers",
                name="Validation Accuracy",
                yaxis="y2"
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Epoch",
            yaxis_title="Loss",
            yaxis2=dict(
                title="Accuracy",
                overlaying="y",
                side="right",
                range=[0, 1]
            ),
            height=540
        )

        return QVizPro._show_fig(fig, show, save_path)

    @staticmethod
    def plot_metric_comparison(
        results,
        title="Model Metric Comparison",
        show=True,
        save_path=None
    ):
        if not isinstance(results, dict):
            raise ValueError("results must be a dictionary")

        model_names = list(results.keys())

        if len(model_names) == 0:
            raise ValueError("results cannot be empty")

        metric_names = list(results[model_names[0]].keys())

        fig = go.Figure()

        for metric in metric_names:
            metric_values = []

            for model in model_names:
                metric_values.append(results[model].get(metric, np.nan))

            fig.add_trace(go.Bar(
                x=model_names,
                y=metric_values,
                name=metric
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Model",
            yaxis_title="Metric Value",
            barmode="group",
            height=560
        )

        return QVizPro._show_fig(fig, show, save_path)

    @staticmethod
    def plot_confusion_matrix(
        y_true,
        y_pred,
        class_names=None,
        title="Confusion Matrix",
        normalize=False,
        show=True,
        save_path=None
    ):
        y_true = QVizPro._to_numpy(y_true).flatten()
        y_pred = QVizPro._to_numpy(y_pred).flatten()

        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")

        matrix = confusion_matrix(y_true, y_pred)

        if normalize:
            row_sums = matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            matrix = matrix / row_sums

        if class_names is None:
            class_names = np.unique(np.concatenate([y_true, y_pred]))

        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=class_names,
            y=class_names,
            colorscale="Blues",
            text=np.round(matrix, 3),
            texttemplate="%{text}",
            colorbar=dict(title="Proportion" if normalize else "Count")
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            height=560
        )

        return QVizPro._show_fig(fig, show, save_path)

    @staticmethod
    def plot_noise_comparison(
        clean_values,
        noisy_values,
        mitigated_values=None,
        title="Noise-Aware QML Comparison",
        show=True,
        save_path=None
    ):
        clean_values = QVizPro._to_numpy(clean_values).flatten()
        noisy_values = QVizPro._to_numpy(noisy_values).flatten()

        if len(clean_values) != len(noisy_values):
            raise ValueError("clean_values and noisy_values must have the same length")

        x_values = np.arange(len(clean_values))

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x_values,
            y=clean_values,
            mode="lines+markers",
            name="Clean"
        ))

        fig.add_trace(go.Scatter(
            x=x_values,
            y=noisy_values,
            mode="lines+markers",
            name="Noisy"
        ))

        if mitigated_values is not None:
            mitigated_values = QVizPro._to_numpy(mitigated_values).flatten()

            if len(mitigated_values) != len(clean_values):
                raise ValueError("mitigated_values must match clean_values")

            fig.add_trace(go.Scatter(
                x=x_values,
                y=mitigated_values,
                mode="lines+markers",
                name="Mitigated"
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Sample / Circuit Index",
            yaxis_title="Output Value",
            height=540
        )

        return QVizPro._show_fig(fig, show, save_path)

    @staticmethod
    def plot_circuit_depths(
        circuit_names,
        depths,
        title="Circuit Depth Comparison",
        show=True,
        save_path=None
    ):
        depths = QVizPro._to_numpy(depths).flatten()

        if len(circuit_names) != len(depths):
            raise ValueError("circuit_names and depths must have the same length")

        fig = go.Figure(data=[
            go.Bar(
                x=circuit_names,
                y=depths,
                name="Circuit Depth"
            )
        ])

        fig.update_layout(
            title=title,
            xaxis_title="Circuit / Model",
            yaxis_title="Depth",
            height=520
        )

        return QVizPro._show_fig(fig, show, save_path)

    @staticmethod
    def plot_probability_distribution(
        probabilities,
        labels=None,
        title="Quantum Measurement Probability Distribution",
        show=True,
        save_path=None
    ):
        probabilities = QVizPro._to_numpy(probabilities).flatten()

        if labels is None:
            num_bits = int(np.ceil(np.log2(len(probabilities))))
            labels = [
                format(i, f"0{num_bits}b")
                for i in range(len(probabilities))
            ]

        if len(labels) != len(probabilities):
            raise ValueError("labels must have the same length as probabilities")

        fig = go.Figure(data=[
            go.Bar(
                x=labels,
                y=probabilities,
                name="Probability"
            )
        ])

        fig.update_layout(
            title=title,
            xaxis_title="Basis State",
            yaxis_title="Probability",
            height=520
        )

        return QVizPro._show_fig(fig, show, save_path)

    @staticmethod
    def plot_research_dashboard(
        losses,
        gradients,
        kernel_matrix=None,
        title="QML Research Dashboard",
        show=True,
        save_path=None
    ):
        losses = QVizPro._to_numpy(losses).flatten()
        gradients = QVizPro._to_numpy(gradients)

        if gradients.ndim == 1:
            gradients = gradients.reshape(1, -1)

        if gradients.ndim != 2:
            raise ValueError("gradients must be a 1D or 2D array")

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Training Loss",
                "Gradient Variance",
                "Gradient Distribution",
                "Kernel Eigenvalues" if kernel_matrix is not None
                else "Mean Absolute Gradient"
            )
        )

        epochs = np.arange(1, len(losses) + 1)
        gradient_variance = np.var(gradients, axis=0)
        gradient_mean = np.mean(np.abs(gradients), axis=0)

        fig.add_trace(go.Scatter(
            x=epochs,
            y=losses,
            mode="lines+markers",
            name="Loss"
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=np.arange(len(gradient_variance)),
            y=gradient_variance,
            mode="lines+markers",
            name="Gradient Variance"
        ), row=1, col=2)

        fig.add_trace(go.Histogram(
            x=gradients.flatten(),
            nbinsx=35,
            name="Gradient Distribution"
        ), row=2, col=1)

        if kernel_matrix is not None:
            kernel_matrix = QVizPro._to_numpy(kernel_matrix)
            kernel_matrix = (kernel_matrix + kernel_matrix.T) / 2

            eigenvalues = np.linalg.eigvalsh(kernel_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]

            fig.add_trace(go.Bar(
                x=np.arange(1, len(eigenvalues) + 1),
                y=eigenvalues,
                name="Kernel Eigenvalues"
            ), row=2, col=2)

        else:
            fig.add_trace(go.Scatter(
                x=np.arange(len(gradient_mean)),
                y=gradient_mean,
                mode="lines+markers",
                name="Mean Absolute Gradient"
            ), row=2, col=2)

        fig.update_layout(
            title=title,
            height=780,
            showlegend=False
        )

        return QVizPro._show_fig(fig, show, save_path)


def plot_hilbert_space(kernel_matrix, labels=None):
    return QVizPro.plot_hilbert_space(kernel_matrix, labels)


def gradient_heatmap(gradients, title="Gradient Magnitude Across Parameters"):
    return QVizPro.gradient_heatmap(gradients, title)


def plot_bloch_sphere(
    state_vectors=None,
    title="Quantum States on Bloch Sphere"
):
    return QVizPro.plot_bloch_sphere(
        state_vectors=state_vectors,
        title=title
    )


def plot_parameter_landscape(
    loss_values,
    title="Loss Landscape in Parameter Space"
):
    return QVizPro.plot_parameter_landscape(
        loss_values,
        title=title
    )